import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.data_processing import CLEAN_OBSERVATION_COLUMNS, save_processed_dataset


def make_clean_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": "Baghdad",
                "country": "Iraq",
                "country_code": "IQ",
                "latitude": 33.34058,
                "longitude": 44.400879,
                "timestamp_utc": "2023-11-14T22:13:20+00:00",
                "aqi": 2,
                "co": 121.6,
                "no": 0.5,
                "no2": 1.25,
                "o3": 80.0,
                "so2": 0.4,
                "pm2_5": 22.5,
                "pm10": 40.0,
                "nh3": 0.75,
            }
        ]
    )


def test_save_processed_dataset_preserves_clean_pollutant_values(tmp_path):
    df = make_clean_dataframe()
    output_file = tmp_path / "clean.csv"

    save_processed_dataset(df, str(output_file))

    saved = pd.read_csv(output_file)
    assert list(saved.columns) == CLEAN_OBSERVATION_COLUMNS
    assert saved.loc[0, "co"] == 121.6
    assert saved.loc[0, "no"] == 0.5
    assert saved.loc[0, "no2"] == 1.25
    assert saved.loc[0, "o3"] == 80.0
    assert saved.loc[0, "so2"] == 0.4
    assert saved.loc[0, "pm2_5"] == 22.5
    assert saved.loc[0, "pm10"] == 40.0
    assert saved.loc[0, "nh3"] == 0.75
    assert saved.loc[0, "city"] == "Baghdad"
    assert saved.loc[0, "country_code"] == "IQ"
