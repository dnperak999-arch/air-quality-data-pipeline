import re

import pandas as pd


REQUIRED_COLUMNS = ["city", "country", "country_code", "latitude", "longitude"]
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")


def validate_locations(
    df: pd.DataFrame,
    expected_row_count: int | None = None,
    forbidden_cities: tuple[str, ...] = (),
) -> None:
    # Check the location table schema and basic geographic constraints.
    # This does not geocode or call an external API.
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if expected_row_count is not None and len(df) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} locations, found {len(df)}."
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Location table contains missing values.")

    latitudes = pd.to_numeric(df["latitude"], errors="coerce")
    longitudes = pd.to_numeric(df["longitude"], errors="coerce")
    if latitudes.isna().any() or longitudes.isna().any():
        raise ValueError("Latitude and longitude must be numeric.")

    if ((latitudes < -90) | (latitudes > 90)).any():
        raise ValueError("Latitude must be between -90 and 90.")

    if ((longitudes < -180) | (longitudes > 180)).any():
        raise ValueError("Longitude must be between -180 and 180.")

    invalid_codes = [
        code
        for code in df["country_code"].astype(str)
        if COUNTRY_CODE_PATTERN.fullmatch(code) is None
    ]
    if invalid_codes:
        raise ValueError(
            "Country codes must be two uppercase letters. "
            f"Invalid values: {sorted(set(invalid_codes))}"
        )

    empty_labels = df["city"].astype(str).str.strip().eq("") | df["country"].astype(str).str.strip().eq("")
    if empty_labels.any():
        raise ValueError("City and country must be non-empty.")

    identity = list(zip(df["city"].astype(str), df["country_code"].astype(str)))
    if len(identity) != len(set(identity)):
        raise ValueError("Duplicate (city, country_code) pairs are not allowed.")

    forbidden = set(forbidden_cities)
    present_forbidden = sorted(set(df["city"].astype(str)) & forbidden)
    if present_forbidden:
        raise ValueError(f"Forbidden city labels are present: {present_forbidden}")
