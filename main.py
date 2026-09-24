from pathlib import Path

from src.data_processing import run_processing_pipeline, save_processed_dataset


DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_LOCATIONS_PATH = "data/worldwide_locations.csv"
DEFAULT_OUTPUT_PATH = "data/processed/air_quality_data.csv"


def main(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    locations_path: str | Path = DEFAULT_LOCATIONS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
):
    # Build the clean dataset from local snapshots and save it.
    clean_df, rejections = run_processing_pipeline(input_dir, locations_path)
    save_processed_dataset(clean_df, str(output_path))

    print(f"Clean observations: {len(clean_df)}")
    print(f"Rejected observations: {len(rejections)}")
    print(f"Processed output: {output_path}")

    if rejections:
        print("Rejections:")
        for rejection in rejections:
            print(f"- {rejection['source_file']}: {rejection['reason']}")

    return clean_df, rejections


if __name__ == "__main__":
    main()
