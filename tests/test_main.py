import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from main import main
from src.data_processing import POLLUTANT_COLUMNS


def write_snapshot(path: Path, latitude: float, longitude: float, co: float) -> None:
    components = {column: 1.0 for column in POLLUTANT_COLUMNS}
    components["co"] = co
    payload = {
        "coord": {"lat": latitude, "lon": longitude},
        "list": [
            {
                "main": {"aqi": 2},
                "components": components,
                "dt": 1700000000,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_clean_csv_without_api_or_scaler(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    locations = tmp_path / "locations.csv"
    output = tmp_path / "processed" / "air_quality_data.csv"

    pd.DataFrame(
        [
            {
                "city": "Baghdad",
                "country": "Iraq",
                "country_code": "IQ",
                "latitude": 33.34058,
                "longitude": 44.400879,
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

    write_snapshot(raw_dir / "Baghdad.json", 33.34058, 44.400879, 121.6)
    write_snapshot(raw_dir / "Paris.json", 10.0, 10.0, 50.0)

    clean_df, rejections = main(raw_dir, locations, output)

    assert output.is_file()
    assert list(tmp_path.rglob("*.pkl")) == []
    assert len(clean_df) == 1
    assert len(rejections) == 1
    assert rejections[0]["source_file"] == "Paris.json"
    assert rejections[0]["reason"] == "coordinate_mismatch"

    saved = pd.read_csv(output)
    assert len(saved) == 1
    assert saved.loc[0, "city"] == "Baghdad"
    assert saved.loc[0, "country"] == "Iraq"
    assert saved.loc[0, "co"] == 121.6
    assert saved.loc[0, "pm2_5"] == 1.0
