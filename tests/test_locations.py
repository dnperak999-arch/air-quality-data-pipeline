import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.locations import validate_locations


LOCATIONS_PATH = PROJECT_ROOT / "data" / "worldwide_locations.csv"

PRESERVED_COORDINATES = {
    "Baghdad": ("33.34058", "44.400879"),
    "Riyadh": ("24.687731", "46.721851"),
    "Tehran": ("35.694389", "51.421509"),
    "Bucharest": ("44.432251", "26.10626"),
    "Sofia": ("42.69751", "23.32415"),
    "Istanbul": ("41.01384", "28.949659"),
    "Warsaw": ("52.229771", "21.01178"),
    "Belgrade": ("44.804008", "20.46513"),
    "Johannesburg": ("-26.202271", "28.043631"),
    "Rawalpindi": ("33.6007", "73.067902"),
    "Faisalabad": ("31.41667", "73.083328"),
    "Dhaka": ("23.7104", "90.40744"),
    "Patna": ("25.6", "85.116669"),
    "New Delhi": ("28.61282", "77.23114"),
    "Kanpur": ("26.466667", "80.349998"),
    "Jodhpur": ("26.286667", "73.029999"),
    "Gaya": ("24.783333", "85"),
    "Delhi": ("28.666668", "77.216667"),
    "Mumbai": ("19.01441", "72.847939"),
    "Agra": ("27.183332", "78.01667"),
    "Kathmandu": ("27.716667", "85.316666"),
    "Omsk": ("55", "73.400002"),
    "Norilsk": ("69.3535", "88.202698"),
    "Kemerovo": ("55.333328", "86.083328"),
    "Bangkok": ("13.87719", "100.71991"),
    "Jakarta": ("-6.21462", "106.845131"),
    "Kuala Lumpur": ("3.14309", "101.686531"),
    "Xi'an": ("34.258331", "108.928612"),
    "Tianjin": ("39.14222", "117.176666"),
    "Lanzhou": ("36.056389", "103.792221"),
    "Incheon": ("37.450001", "126.416107"),
    "Ulan-Ude": ("51.82605", "107.609787"),
    "Shenyang": ("41.792221", "123.432777"),
    "Sydney": ("-33.867851", "151.207321"),
    "Bamenda": ("5.95266", "10.15824"),
    "Lisbon": ("38.716671", "-9.13333"),
    "Stockholm": ("59.5", "18"),
    "Paris": ("48.853401", "2.3486"),
    "Marseille": ("43.296951", "5.38107"),
    "Prague": ("50.088039", "14.42076"),
    "Katowice": ("50.258419", "19.02754"),
    "Vicenza": ("45.557289", "11.5409"),
    "Turin": ("45.070492", "7.68682"),
    "Piacenza": ("45.046761", "9.69937"),
    "Slavonski Brod": ("45.160278", "18.01556"),
    "Cape Town": ("-33.925838", "18.42322"),
    "Toluca": ("19.28833", "-99.667221"),
    "Mexico City": ("19.428471", "-99.127663"),
}

CORRECTED_LOCATIONS = {
    "Wellington": ("New Zealand", "NZ", -41.287, 174.776),
    "Santiago": ("Chile", "CL", -33.45694, -70.64827),
    "Madrid": ("Spain", "ES", 40.41650, -3.70256),
    "Los Angeles": ("United States", "US", 34.05223, -118.24368),
    "Denver": ("United States", "US", 39.73915, -104.98470),
    "Chicago": ("United States", "US", 41.85003, -87.65005),
    "Shijiazhuang": ("China", "CN", 38.04250, 114.51000),
    "Buenos Aires": ("Argentina", "AR", -34.61315, -58.37723),
    "Lima": ("Peru", "PE", -12.06000, -77.03750),
    "Banff": ("Canada", "CA", 51.17622, -115.56982),
    "Vancouver": ("Canada", "CA", 49.24966, -123.11934),
}


def read_location_rows() -> list[dict[str, str]]:
    with LOCATIONS_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_locations_frame() -> pd.DataFrame:
    return pd.read_csv(LOCATIONS_PATH)


def test_location_file_has_required_schema_and_59_rows():
    rows = read_location_rows()
    frame = load_locations_frame()

    assert list(frame.columns) == [
        "city",
        "country",
        "country_code",
        "latitude",
        "longitude",
    ]
    assert len(rows) == 59
    assert len(frame) == 59
    validate_locations(frame, expected_row_count=59, forbidden_cities=("California",))


def test_preserved_coordinates_are_unchanged():
    rows = {row["city"]: row for row in read_location_rows()}

    assert len(PRESERVED_COORDINATES) == 48
    for city, (latitude, longitude) in PRESERVED_COORDINATES.items():
        assert rows[city]["latitude"] == latitude
        assert rows[city]["longitude"] == longitude


def test_corrected_locations_match_approved_places():
    frame = load_locations_frame().set_index("city")

    for city, (country, country_code, latitude, longitude) in CORRECTED_LOCATIONS.items():
        row = frame.loc[city]
        assert row["country"] == country
        assert row["country_code"] == country_code
        assert row["latitude"] == pytest.approx(latitude)
        assert row["longitude"] == pytest.approx(longitude)

    wellington = frame.loc["Wellington"]
    assert wellington["latitude"] < 0
    assert wellington["longitude"] > 170

    santiago = frame.loc["Santiago"]
    assert santiago["latitude"] < 0
    assert santiago["longitude"] == pytest.approx(-70.64827, abs=1)

    madrid = frame.loc["Madrid"]
    assert madrid["latitude"] == pytest.approx(40.4, abs=1)
    assert madrid["longitude"] == pytest.approx(-3.7, abs=0.1)

    assert frame.loc["Los Angeles", "longitude"] < -118
    assert frame.loc["Denver", "latitude"] == pytest.approx(39.7, abs=1)
    assert frame.loc["Denver", "longitude"] == pytest.approx(-105, abs=1)
    assert frame.loc["Chicago", "latitude"] > 40
    assert frame.loc["Chicago", "longitude"] == pytest.approx(-87.65, abs=0.2)
    assert frame.loc["Shijiazhuang", "longitude"] > 114
    assert frame.loc["Buenos Aires", "latitude"] == pytest.approx(-34.6, abs=0.1)
    assert frame.loc["Buenos Aires", "longitude"] == pytest.approx(-58.4, abs=0.1)
    assert frame.loc["Lima", "longitude"] < -77


def test_california_row_is_absent():
    cities = [row["city"] for row in read_location_rows()]
    assert "California" not in cities


def test_validate_locations_rejects_duplicate_identity():
    frame = pd.DataFrame(
        [
            {
                "city": "Banff",
                "country": "Canada",
                "country_code": "CA",
                "latitude": 51.17622,
                "longitude": -115.56982,
            },
            {
                "city": "Banff",
                "country": "Canada",
                "country_code": "CA",
                "latitude": 51.2,
                "longitude": -115.6,
            },
        ]
    )

    with pytest.raises(ValueError, match="Duplicate"):
        validate_locations(frame)


def test_fetch_reads_city_column_without_calling_network(tmp_path):
    from src.data_collection import fetch_and_save_data

    class FakeClient:
        def get_air_pollution_data(self, lat, lon):
            return {"coord": {"lat": lat, "lon": lon}}

    frame = pd.DataFrame(
        [
            {
                "city": "Test City",
                "country": "Canada",
                "country_code": "CA",
                "latitude": 51.17622,
                "longitude": -115.56982,
            }
        ]
    )

    fetch_and_save_data(frame, FakeClient(), output_dir=str(tmp_path))

    assert (tmp_path / "Test_City.json").exists()
    assert not (PROJECT_ROOT / "data" / "raw" / "Test_City.json").exists()
