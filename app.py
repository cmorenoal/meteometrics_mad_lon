from pathlib import Path

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
    initial_records = len(all_cities_data)

    # 1. Sustituye comas decimales por puntos en las columnas meteorológicas.
    metric_columns = all_cities_data.filter(regex=r"^metrics\.").columns
    all_cities_data[metric_columns] = all_cities_data[metric_columns].replace(
        ",", ".", regex=True
    )

    # 2. Excluye la ciudad y las unidades; las demás columnas contienen números.
    unit_columns = [
        "metrics.temperature.unit",
        "metrics.precipitation.unit",
    ]
    numeric_columns = all_cities_data.drop(
        columns=["city", *unit_columns]
    ).columns

    # 3. Convierte texto numérico; los valores no válidos pasan a nulos.
    all_cities_data[numeric_columns] = all_cities_data[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # 4. Limpia explícitamente el valor centinela.
    all_cities_data[numeric_columns] = all_cities_data[numeric_columns].replace(
        -999.0, float("nan")
    )

    # 5. Conserva un registro por ciudad, año y mes.
    all_cities_data = all_cities_data.drop_duplicates(
        subset=["city", "year", "month"]
    )

    # 6. Ordena cronológicamente dentro de cada ciudad.
    all_cities_data = all_cities_data.sort_values(
        ["city", "year", "month"], ignore_index=True
    )

    # 7. Convierte temperaturas y precipitaciones según sus unidades.
    temperature_columns = [
        "metrics.temperature.max",
        "metrics.temperature.avg",
        "metrics.temperature.min",
    ]
    fahrenheit_rows = all_cities_data["metrics.temperature.unit"].eq("F")
    all_cities_data.loc[fahrenheit_rows, temperature_columns] = (
        all_cities_data.loc[fahrenheit_rows, temperature_columns] - 32
    ) * (5 / 9)
    all_cities_data.loc[fahrenheit_rows, "metrics.temperature.unit"] = "C"

    inches_rows = all_cities_data["metrics.precipitation.unit"].eq("inches")
    all_cities_data.loc[inches_rows, "metrics.precipitation.avg"] *= 25.4
    all_cities_data.loc[inches_rows, "metrics.precipitation.unit"] = "mm"

    # 8. Las conversiones usan máscaras vectorizadas, sin recorrer filas.
    deleted_records = initial_records - len(all_cities_data)
    report_path = Path(__file__).with_name("informe_limpieza.txt")
    report_path.write_text(
        f"Registros iniciales: {initial_records}\n"
        f"Registros borrados: {deleted_records}\n"
        f"Registros finales: {len(all_cities_data)}\n",
        encoding="utf-8",
    )

    print(all_cities_data.head())

except requests.exceptions.RequestException as e:
    print(f'Error en la petición: {e}')
