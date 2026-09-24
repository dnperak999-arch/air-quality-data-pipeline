import requests


class OpenWeatherAirPollution:
    # Simple client used to get air pollution data from OpenWeather

    BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution"

    def __init__(self, api_key: str):
        # Store API key for later requests
        if not api_key or not isinstance(api_key, str):
            raise ValueError("A valid API key must be provided.")

        self.api_key = api_key

    def get_air_pollution_data(self, lat: float, lon: float) -> dict:
        # Request air pollution data for a given latitude and longitude
        if not isinstance(lat, (int, float)):
            raise ValueError("Latitude must be numeric.")

        if not isinstance(lon, (int, float)):
            raise ValueError("Longitude must be numeric.")
        # Parameters required by the API
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }
        # Send the request to the API
        response = requests.get(self.BASE_URL, params=params, timeout=15)
        response.raise_for_status()

        # Convert the JSON response into a Python dictionary
        return response.json()