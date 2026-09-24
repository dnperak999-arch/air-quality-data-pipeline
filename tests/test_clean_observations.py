import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing import (
    CLEAN_OBSERVATION_COLUMNS,
    POLLUTANT_COLUMNS,
    REASON_COORDINATE_MISMATCH,
    REASON_DUPLICATE_OBSERVATION,
    REASON_INVALID_AQI,
    REASON_INVALID_TIMESTAMP,
    REASON_MALFORMED_OBSERVATION,
    REASON_MISSING_POLLUTANT,
    REASON_UNKNOWN_LOCATION,
    build_clean_observations,
    run_processing_pipeline,
)


CANONICAL_LATITUDE = 33.34058
CANONICAL_LONGITUDE = 44.400879
KNOWN_DT = 1700000000
KNOWN_TIMESTAMP = "2023-11-14T22:13:20+00:00"


def write_locations(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "city": "Baghdad",
                "country": "Iraq",
                "country_code": "IQ",
                "latitude": CANONICAL_LATITUDE,
                "longitude": CANONICAL_LONGITUDE,
            }
        ]
    )
    frame.to_csv(path, index=False)


def write_snapshot(
    path: Path,
    latitude: float = CANONICAL_LATITUDE,
    longitude: float = CANONICAL_LONGITUDE,
    dt: object = KNOWN_DT,
    aqi: object = 2,
    components: dict | None = None,
    include_components: bool = True,
) -> None:
    if components is None and include_components:
        components = {column: 1.5 for column in POLLUTANT_COLUMNS}
        components["co"] = 121.6
        components["pm2_5"] = 22.5

    entry = {
        "main": {"aqi": aqi},
        "dt": dt,
    }
    if include_components:
        entry["components"] = components

    payload = {
        "coord": {"lat": latitude, "lon": longitude},
        "list": [entry],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_observation_keeps_original_pollutants_and_country(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    write_snapshot(snapshot)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert rejections == []
    assert list(clean.columns) == CLEAN_OBSERVATION_COLUMNS
    assert len(clean) == 1
    row = clean.iloc[0]
    assert row["city"] == "Baghdad"
    assert row["country"] == "Iraq"
    assert row["country_code"] == "IQ"
    assert row["latitude"] == CANONICAL_LATITUDE
    assert row["longitude"] == CANONICAL_LONGITUDE
    assert row["co"] == 121.6
    assert row["pm2_5"] == 22.5
    assert row["no"] == 1.5


def test_known_unix_timestamp_converts_to_utc(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    write_snapshot(snapshot, dt=KNOWN_DT)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert rejections == []
    assert clean.loc[0, "timestamp_utc"] == KNOWN_TIMESTAMP
    assert str(clean.loc[0, "timestamp_utc"]).endswith("+00:00")
    assert "dt" not in clean.columns


def test_coordinate_difference_within_tolerance_is_accepted(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    observed_latitude = CANONICAL_LATITUDE + 0.01
    observed_longitude = CANONICAL_LONGITUDE - 0.01
    write_snapshot(snapshot, latitude=observed_latitude, longitude=observed_longitude)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert rejections == []
    assert len(clean) == 1
    assert clean.loc[0, "latitude"] == observed_latitude
    assert clean.loc[0, "longitude"] == observed_longitude
    assert clean.loc[0, "country_code"] == "IQ"


def test_displaced_coordinates_are_rejected_without_relabelling(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    write_snapshot(snapshot, latitude=10.0, longitude=10.0)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert len(rejections) == 1
    rejection = rejections[0]
    assert rejection["source_file"] == "Baghdad.json"
    assert rejection["reason"] == REASON_COORDINATE_MISMATCH
    assert rejection["raw_latitude"] == 10.0
    assert rejection["raw_longitude"] == 10.0
    assert rejection["canonical_latitude"] == CANONICAL_LATITUDE
    assert rejection["canonical_longitude"] == CANONICAL_LONGITUDE
    assert list(clean.columns) == CLEAN_OBSERVATION_COLUMNS


def test_unknown_location_is_rejected_instead_of_nearest_match(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "California.json"
    write_locations(locations)
    write_snapshot(snapshot, latitude=CANONICAL_LATITUDE, longitude=CANONICAL_LONGITUDE)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert len(rejections) == 1
    assert rejections[0]["source_file"] == "California.json"
    assert rejections[0]["reason"] == REASON_UNKNOWN_LOCATION
    assert "canonical_latitude" not in rejections[0]


def test_missing_pollutant_is_rejected_without_imputation(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    components = {column: 1.5 for column in POLLUTANT_COLUMNS}
    components["co"] = None
    write_snapshot(snapshot, components=components)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert len(rejections) == 1
    assert rejections[0]["reason"] == REASON_MISSING_POLLUTANT
    assert "co" not in clean.columns or clean.empty


def test_invalid_aqi_is_rejected(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    write_snapshot(snapshot, aqi=9)

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert rejections[0]["reason"] == REASON_INVALID_AQI


def test_invalid_timestamp_is_rejected(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    write_snapshot(snapshot, dt="not-a-timestamp")

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert rejections[0]["reason"] == REASON_INVALID_TIMESTAMP


def test_duplicate_observation_is_not_returned_twice(tmp_path):
    locations = tmp_path / "locations.csv"
    write_locations(locations)
    first = tmp_path / "first" / "Baghdad.json"
    second = tmp_path / "second" / "Baghdad.json"
    first.parent.mkdir()
    second.parent.mkdir()
    write_snapshot(first)
    write_snapshot(second)

    clean, rejections = build_clean_observations([first, second], locations)

    assert len(clean) == 1
    assert clean.loc[0, "city"] == "Baghdad"
    assert len(rejections) == 1
    assert rejections[0]["source_file"] == "Baghdad.json"
    assert rejections[0]["reason"] == REASON_DUPLICATE_OBSERVATION


def test_run_processing_pipeline_keeps_original_values_and_rejects_mismatches(tmp_path):
    locations = tmp_path / "locations.csv"
    pd.DataFrame(
        [
            {
                "city": "Baghdad",
                "country": "Iraq",
                "country_code": "IQ",
                "latitude": CANONICAL_LATITUDE,
                "longitude": CANONICAL_LONGITUDE,
            },
            {
                "city": "Paris",
                "country": "France",
                "country_code": "FR",
                "latitude": 48.853401,
                "longitude": 2.3486,
            },
        ]
    ).to_csv(locations, index=False)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_snapshot(raw_dir / "Baghdad.json")
    write_snapshot(raw_dir / "Paris.json", latitude=10.0, longitude=10.0)

    clean, rejections = run_processing_pipeline(raw_dir, locations)

    assert list(clean.columns) == CLEAN_OBSERVATION_COLUMNS
    assert len(clean) == 1
    assert clean.loc[0, "city"] == "Baghdad"
    assert clean.loc[0, "country"] == "Iraq"
    assert clean.loc[0, "country_code"] == "IQ"
    assert clean.loc[0, "co"] == 121.6
    assert clean.loc[0, "pm2_5"] == 22.5
    assert clean.loc[0, "timestamp_utc"] == KNOWN_TIMESTAMP
    assert "Paris" not in set(clean["city"])
    assert len(rejections) == 1
    assert rejections[0]["source_file"] == "Paris.json"
    assert rejections[0]["reason"] == REASON_COORDINATE_MISMATCH
    assert list(tmp_path.rglob("*.pkl")) == []


def test_malformed_observation_is_rejected(tmp_path):
    locations = tmp_path / "locations.csv"
    snapshot = tmp_path / "Baghdad.json"
    write_locations(locations)
    snapshot.write_text(json.dumps({"list": []}), encoding="utf-8")

    clean, rejections = build_clean_observations([snapshot], locations)

    assert clean.empty
    assert rejections[0]["reason"] == REASON_MALFORMED_OBSERVATION
