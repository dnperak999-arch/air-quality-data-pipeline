from pathlib import Path


def load_api_key(file_name: str = "apikey.txt") -> str:
    # Load API key from a local file (kept outside the code for security)
    # This avoids hardcoding the key directly in the source code

    project_root = Path(__file__).resolve().parent
    key_path = project_root / "config" / file_name

    if not key_path.exists():
        raise FileNotFoundError(f"API key file was not found: config/{file_name}")

    api_key = key_path.read_text(encoding="utf-8").strip()

    if not api_key:
        raise ValueError("API key file is empty.")

    return api_key