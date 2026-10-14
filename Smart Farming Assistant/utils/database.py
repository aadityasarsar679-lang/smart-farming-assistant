"""SQLite storage and password-safe account helpers for Smart Farming Assistant."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "smart_farming.db"
PBKDF2_ITERATIONS = 310_000


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    """Create tables automatically the first time the app starts."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                module TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                predicted_value TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                due_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()
    return digest, salt.hex()


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    email = email.strip().lower()

    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
        return False, "Username 3–30 characters ka ho aur letters, numbers, dot, underscore ya hyphen use kare."
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return False, "Valid email address enter karo."
    if len(password) < 8:
        return False, "Password kam se kam 8 characters ka hona chahiye."

    password_hash, salt = _hash_password(password)
    try:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, salt),
            )
        return True, "Account ban gaya. Ab Login tab se sign in karo."
    except sqlite3.IntegrityError:
        return False, "Yeh username ya email pehle se registered hai. Doosra use karo."
    except sqlite3.Error:
        return False, "Database mein account save nahi ho saka. App restart karke dobara try karo."


def authenticate_user(identifier: str, password: str) -> dict[str, Any] | None:
    identifier = identifier.strip()
    try:
        with get_connection() as connection:
            row = connection.execute(
                """SELECT id, username, email, password_hash, salt
                   FROM users WHERE username = ? OR email = ? LIMIT 1""",
                (identifier, identifier),
            ).fetchone()
        if row is None:
            return None
        attempted_hash, _ = _hash_password(password, row["salt"])
        if not hmac.compare_digest(attempted_hash, row["password_hash"]):
            return None
        return {"id": int(row["id"]), "username": row["username"], "email": row["email"]}
    except sqlite3.Error:
        return None


def save_prediction(
    user_id: int,
    module: str,
    dataset_name: str,
    target_name: str,
    predicted_value: Any,
    inputs: dict[str, Any],
) -> None:
    """Save one prediction under the logged-in user's ID."""
    with get_connection() as connection:
        connection.execute(
            """INSERT INTO prediction_history
               (user_id, module, dataset_name, target_name, predicted_value, inputs_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                int(user_id),
                str(module),
                str(dataset_name),
                str(target_name),
                str(predicted_value),
                json.dumps(inputs, ensure_ascii=False, default=str),
            ),
        )


def get_prediction_history(user_id: int, limit: int = 500) -> list[dict[str, Any]]:
    """Only return records belonging to the requested signed-in user."""
    with get_connection() as connection:
        rows = connection.execute(
            """SELECT id, module, dataset_name, target_name, predicted_value,
                      inputs_json, created_at
               FROM prediction_history WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (int(user_id), max(1, min(int(limit), 2000))),
        ).fetchall()
    return [dict(row) for row in rows]


def create_reminder(user_id: int, title: str, description: str, due_at: str) -> int:
    """Create a reminder owned by one user. due_at format: YYYY-MM-DD HH:MM (IST)."""
    title = title.strip()
    description = description.strip()
    if not title:
        raise ValueError("Reminder title is required.")
    with get_connection() as connection:
        cursor = connection.execute(
            """INSERT INTO reminders (user_id, title, description, due_at, status)
               VALUES (?, ?, ?, ?, 'pending')""",
            (int(user_id), title, description, due_at),
        )
        return int(cursor.lastrowid)


def get_user_reminders(user_id: int, include_completed: bool = False) -> list[dict[str, Any]]:
    """Return only the signed-in user's reminders."""
    query = """SELECT id, user_id, title, description, due_at, status, created_at
               FROM reminders WHERE user_id = ?"""
    params: list[Any] = [int(user_id)]
    if not include_completed:
        query += " AND status != 'completed'"
    query += " ORDER BY due_at ASC, id DESC"
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def complete_reminder(user_id: int, reminder_id: int) -> bool:
    """Mark one of the user's reminders completed."""
    with get_connection() as connection:
        cursor = connection.execute(
            """UPDATE reminders SET status = 'completed'
               WHERE id = ? AND user_id = ? AND status != 'completed'""",
            (int(reminder_id), int(user_id)),
        )
        return cursor.rowcount > 0


def delete_reminder(user_id: int, reminder_id: int) -> bool:
    """Delete one reminder, constrained to its owner."""
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM reminders WHERE id = ? AND user_id = ?",
            (int(reminder_id), int(user_id)),
        )
        return cursor.rowcount > 0


def process_due_reminders(user_id: int, now_local: str) -> list[str]:
    """Create a notification once when each pending reminder becomes due.

    now_local must be formatted YYYY-MM-DD HH:MM in Asia/Kolkata time. The
    reminder status becomes 'notified', preventing duplicate notifications.
    """
    newly_due: list[str] = []
    with get_connection() as connection:
        rows = connection.execute(
            """SELECT id, title, description, due_at FROM reminders
               WHERE user_id = ? AND status = 'pending' AND due_at <= ?
               ORDER BY due_at ASC""",
            (int(user_id), now_local),
        ).fetchall()
        for row in rows:
            message = f"Reminder due: {row['title']}"
            if row["description"]:
                message += f" — {row['description']}"
            connection.execute(
                "INSERT INTO notifications (user_id, kind, message) VALUES (?, 'reminder', ?)",
                (int(user_id), message),
            )
            connection.execute(
                "UPDATE reminders SET status = 'notified' WHERE id = ? AND user_id = ? AND status = 'pending'",
                (int(row["id"]), int(user_id)),
            )
            newly_due.append(str(row["title"]))
    return newly_due


def get_user_notifications(
    user_id: int, unread_only: bool = False, limit: int = 100
) -> list[dict[str, Any]]:
    """Return only the signed-in user's notifications."""
    query = """SELECT id, user_id, kind, message, is_read, created_at
               FROM notifications WHERE user_id = ?"""
    params: list[Any] = [int(user_id)]
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 500)))
    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_unread_notification_count(user_id: int) -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM notifications WHERE user_id = ? AND is_read = 0",
            (int(user_id),),
        ).fetchone()
    return int(row["count"] if row else 0)


def mark_notification_read(user_id: int, notification_id: int) -> bool:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?",
            (int(notification_id), int(user_id)),
        )
        return cursor.rowcount > 0


def mark_all_notifications_read(user_id: int) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE notifications SET is_read = 1 WHERE user_id = ? AND is_read = 0",
            (int(user_id),),
        )
        return int(cursor.rowcount)
