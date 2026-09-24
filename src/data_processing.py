import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


POLLUTANT_COLUMNS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
DEFAULT_PROCESSED_PATH = "data/processed/air_quality_data.csv"
DEFAULT_LOCATIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "worldwide_locations.csv"
COORDINATE_TOLERANCE_DEGREES = 0.02

CLEAN_OBSERVATION_COLUMNS = [
    "city",
    "country",
    "country_code",
    "latitude",
    "longitude",
    "timestamp_utc",
    "aqi",
    *POLLUTANT_COLUMNS,
]

REASON_UNKNOWN_LOCATION = "unknown_location"
REASON_COORDINATE_MISMATCH = "coordinate_mismatch"
REASON_MALFORMED_OBSERVATION = "malformed_observation"
REASON_MISSING_POLLUTANT = "missing_pollutant"
REASON_INVALID_AQI = "invalid_aqi"
REASON_INVALID_TIMESTAMP = "invalid_timestamp"
REASON_DUPLICATE_OBSERVATION = "duplicate_observation"


def run_processing_pipeline(
    input_dir: str | Path = "data/raw",
    locations_path: str | Path = DEFAULT_LOCATIONS_PATH,
) -> tuple[pd.DataFrame, list[dict]]:
    # Validate raw snapshots against the canonical location table.
    # Pollutant concentrations are copied through unchanged.
    json_paths = sorted(Path(input_dir).glob("*.json"))
    return build_clean_observations(json_paths, locations_path)


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: str = DEFAULT_PROCESSED_PATH,
) -> None:
    # Write the clean DataFrame without transforming its values.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_clean_observations(
    json_paths: list[Path],
    locations_path: str | Path = DEFAULT_LOCATIONS_PATH,
) -> tuple[pd.DataFrame, list[dict]]:
    # Parse raw OpenWeather snapshots into original-concentration observations.
    # This path does not scale or impute pollutant values.
    locations = pd.read_csv(locations_path)
    accepted: list[dict] = []
    rejections: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for json_path in json_paths:
        observation, rejection = _parse_clean_observation(Path(json_path), locations)
        if rejection is not None:
            rejections.append(rejection)
            continue

        identity = (
            observation["city"],
            observation["country_code"],
            observation["timestamp_utc"],
        )
        if identity in seen_keys:
            rejections.append(
                _rejection(
                    Path(json_path).name,
                    REASON_DUPLICATE_OBSERVATION,
                    raw_latitude=observation["latitude"],
                    raw_longitude=observation["longitude"],
                    canonical_latitude=_canonical_value(locations, observation["city"], "latitude"),
                    canonical_longitude=_canonical_value(locations, observation["city"], "longitude"),
                )
            )
            continue

        seen_keys.add(identity)
        accepted.append(observation)

    frame = pd.DataFrame.from_records(accepted, columns=CLEAN_OBSERVATION_COLUMNS)
    return frame, rejections


def _parse_clean_observation(
    path: Path,
    locations: pd.DataFrame,
) -> tuple[dict | None, dict | None]:
    source_file = path.name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, _rejection(source_file, REASON_MALFORMED_OBSERVATION)

    if not isinstance(payload, dict):
        return None, _rejection(source_file, REASON_MALFORMED_OBSERVATION)

    raw_latitude, raw_longitude = _raw_coordinates(payload)
    entry = _first_entry(payload)
    if entry is None or raw_latitude is None or raw_longitude is None:
        return None, _rejection(
            source_file,
            REASON_MALFORMED_OBSERVATION,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
        )

    if not _coordinates_in_range(raw_latitude, raw_longitude):
        return None, _rejection(
            source_file,
            REASON_MALFORMED_OBSERVATION,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
        )

    city = path.stem.replace("_", " ")
    matches = locations.loc[locations["city"] == city]
    if matches.empty:
        return None, _rejection(
            source_file,
            REASON_UNKNOWN_LOCATION,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
        )

    canonical = matches.iloc[0]
    canonical_latitude = float(canonical["latitude"])
    canonical_longitude = float(canonical["longitude"])
    if (
        abs(raw_latitude - canonical_latitude) > COORDINATE_TOLERANCE_DEGREES
        or abs(raw_longitude - canonical_longitude) > COORDINATE_TOLERANCE_DEGREES
    ):
        return None, _rejection(
            source_file,
            REASON_COORDINATE_MISMATCH,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
            canonical_latitude=canonical_latitude,
            canonical_longitude=canonical_longitude,
        )

    timestamp_utc = _utc_timestamp(entry.get("dt"))
    if timestamp_utc is None:
        return None, _rejection(
            source_file,
            REASON_INVALID_TIMESTAMP,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
            canonical_latitude=canonical_latitude,
            canonical_longitude=canonical_longitude,
        )

    main = entry.get("main")
    aqi = main.get("aqi") if isinstance(main, dict) else None
    if type(aqi) is not int or not 1 <= aqi <= 5:
        return None, _rejection(
            source_file,
            REASON_INVALID_AQI,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
            canonical_latitude=canonical_latitude,
            canonical_longitude=canonical_longitude,
        )

    components = entry.get("components")
    if not isinstance(components, dict):
        return None, _rejection(
            source_file,
            REASON_MISSING_POLLUTANT,
            raw_latitude=raw_latitude,
            raw_longitude=raw_longitude,
            canonical_latitude=canonical_latitude,
            canonical_longitude=canonical_longitude,
        )

    pollutants = {}
    for column in POLLUTANT_COLUMNS:
        if column not in components or components[column] is None:
            return None, _rejection(
                source_file,
                REASON_MISSING_POLLUTANT,
                raw_latitude=raw_latitude,
                raw_longitude=raw_longitude,
                canonical_latitude=canonical_latitude,
                canonical_longitude=canonical_longitude,
            )
        value = components[column]
        if not _is_finite_number(value):
            return None, _rejection(
                source_file,
                REASON_MISSING_POLLUTANT,
                raw_latitude=raw_latitude,
                raw_longitude=raw_longitude,
                canonical_latitude=canonical_latitude,
                canonical_longitude=canonical_longitude,
            )
        pollutants[column] = value

    observation = {
        "city": city,
        "country": canonical["country"],
        "country_code": canonical["country_code"],
        "latitude": raw_latitude,
        "longitude": raw_longitude,
        "timestamp_utc": timestamp_utc,
        "aqi": aqi,
        **pollutants,
    }
    return observation, None


def _first_entry(payload: dict) -> dict | None:
    entries = payload.get("list")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return None
    return entries[0]


def _raw_coordinates(payload: dict) -> tuple[float | None, float | None]:
    coord = payload.get("coord")
    if not isinstance(coord, dict):
        return None, None

    latitude = coord.get("lat")
    longitude = coord.get("lon")
    if not _is_finite_number(latitude) or not _is_finite_number(longitude):
        return None, None
    return latitude, longitude


def _coordinates_in_range(latitude: float, longitude: float) -> bool:
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def _utc_timestamp(dt: object) -> str | None:
    if not _is_finite_number(dt):
        return None

    try:
        return datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _canonical_value(locations: pd.DataFrame, city: str, column: str) -> float | None:
    matches = locations.loc[locations["city"] == city]
    if matches.empty:
        return None
    return float(matches.iloc[0][column])


def _rejection(
    source_file: str,
    reason: str,
    raw_latitude: float | None = None,
    raw_longitude: float | None = None,
    canonical_latitude: float | None = None,
    canonical_longitude: float | None = None,
) -> dict:
    record = {
        "source_file": source_file,
        "reason": reason,
    }
    if raw_latitude is not None:
        record["raw_latitude"] = raw_latitude
    if raw_longitude is not None:
        record["raw_longitude"] = raw_longitude
    if canonical_latitude is not None:
        record["canonical_latitude"] = canonical_latitude
    if canonical_longitude is not None:
        record["canonical_longitude"] = canonical_longitude
    return record