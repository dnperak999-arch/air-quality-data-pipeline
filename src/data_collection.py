import pandas as pd
import json
from pathlib import Path


def load_locations(file_path: str) -> pd.DataFrame:
    # Load locations dataset from CSV file and display basic info

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Columns:", df.columns.tolist())
    print("\nFirst rows:")
    print(df.head())

    return df


def fetch_and_save_data(df, client, output_dir="data/raw"):
    # Fetch air pollution data for each location and save it as JSON files

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through each location in the dataset
    for index, row in df.iterrows():
        city = row["city"]
        lat = row["latitude"]
        lon = row["longitude"]

        print(f"Fetching data for {city}...")

        # Try to request data from the API and save it locally
        try:
            data = client.get_air_pollution_data(lat, lon)

            file_name = f"{city.replace(' ', '_')}.json"
            file_path = output_path / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

        # Handle errors (e.g., API issues or invalid data)
        except Exception as e:
            print(f"Error for {city}: {e}")
