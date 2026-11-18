import pandas as pd
import requests

urls = [
    "https://meteometrics.rlopez.org/historical-data/MAD",
    "https://meteometrics.rlopez.org/historical-data/LON",
]

try:
    city_dataframes = []

    for url in urls:
        response = requests.get(url)
        response.raise_for_status()

        city_data = pd.json_normalize(response.json())
        city_data["city"] = url[-3:]
        city_dataframes.append(city_data)

    all_cities_data = pd.concat(city_dataframes, ignore_index=True)
    print(all_cities_data.head())

except requests.exceptions.RequestException as e:
    print(f'Error en la petición: {e}')
