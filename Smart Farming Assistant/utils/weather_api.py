"""Small Open-Meteo client used by Smart Farming Assistant.

Open-Meteo's public API is intended for free non-commercial use. Check its
current terms before using this project commercially.
"""
from __future__ import annotations

from typing import Any

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def search_locations(city: str) -> list[dict[str, Any]]:
    """Find Indian locations matching a city/town name."""
    city = city.strip()
    if len(city) < 2:
        raise ValueError("City ka naam kam se kam 2 characters ka hona chahiye.")
    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 10, "language": "en", "format": "json", "countryCode": "IN"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Location search failed."))
    return payload.get("results", []) or []


def get_weather_forecast(latitude: float, longitude: float) -> dict[str, Any]:
    """Return current conditions and a 7-day forecast for coordinates."""
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join([
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m",
            ]),
            "daily": ",".join([
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "precipitation_probability_max", "wind_speed_10m_max",
            ]),
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(payload.get("reason", "Weather request failed."))
    return payload


def weather_code_label(code: Any) -> str:
    """Readable descriptions for WMO weather codes used by Open-Meteo."""
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Condition unavailable"

    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
        55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain",
        67: "Heavy freezing rain", 71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Weather code {code}")
