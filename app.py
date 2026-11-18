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

    df_total = pd.concat(city_dataframes, ignore_index=True)
    initial_records = len(df_total)

    # 1. Sustituye comas decimales por puntos en las columnas meteorológicas.
    metric_columns = df_total.filter(regex=r"^metrics\.").columns
    df_total[metric_columns] = df_total[metric_columns].replace(
        ",", ".", regex=True
    )

    # 2. Excluye la ciudad y las unidades; las demás columnas contienen números.
    unit_columns = [
        "metrics.temperature.unit",
        "metrics.precipitation.unit",
    ]
    numeric_columns = df_total.drop(
        columns=["city", *unit_columns]
    ).columns

    # 3. Convierte texto numérico; los valores no válidos pasan a nulos.
    df_total[numeric_columns] = df_total[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # 4. Limpia explícitamente el valor centinela.
    df_total[numeric_columns] = df_total[numeric_columns].replace(
        -999.0, float("nan")
    )

    # 5. Conserva un registro por ciudad, año y mes.
    df_total = df_total.drop_duplicates(
        subset=["city", "year", "month"]
    )

    # 6. Ordena cronológicamente dentro de cada ciudad.
    df_total = df_total.sort_values(
        ["city", "year", "month"], ignore_index=True
    )

    # 7. Convierte temperaturas y precipitaciones según sus unidades.
    temperature_columns = [
        "metrics.temperature.max",
        "metrics.temperature.avg",
        "metrics.temperature.min",
    ]
    fahrenheit_rows = df_total["metrics.temperature.unit"].eq("F")
    df_total.loc[fahrenheit_rows, temperature_columns] = (
        df_total.loc[fahrenheit_rows, temperature_columns] - 32
    ) * (5 / 9)
    df_total.loc[fahrenheit_rows, "metrics.temperature.unit"] = "C"

    inches_rows = df_total["metrics.precipitation.unit"].eq("inches")
    df_total.loc[inches_rows, "metrics.precipitation.avg"] *= 25.4
    df_total.loc[inches_rows, "metrics.precipitation.unit"] = "mm"

    # 8. Calcula el enriquecimiento de forma vectorizada, sin recorrer filas.
    # 9. Añade variables y KPI al DataFrame total.
    # La oscilación térmica mensual es la diferencia entre la máxima y la mínima.
    df_total["thermal_range"] = (
        df_total["metrics.temperature.max"] - df_total["metrics.temperature.min"]
    )

    # cumsum acumula la lluvia y reinicia el total para cada ciudad y año.
    # El orden cronológico por ciudad, año y mes se estableció en el punto 6.
    df_total["cumulative_rain_year"] = df_total.groupby(
        ["city", "year"]
    )["metrics.precipitation.avg"].cumsum()

    # El índice aumenta con temperaturas máximas altas y lluvia media baja.
    # La lluvia está expresada en milímetros; se suma 1 para evitar dividir por cero.
    df_total["kpi.heat_stress_index"] = df_total[
        "metrics.temperature.max"
    ] / (1 + df_total["metrics.precipitation.avg"])

    # Compara cada media mensual con el histórico de su ciudad en desviaciones estándar.
    # transform devuelve medias y desviaciones alineadas con cada fila del DataFrame.
    city_temperature = df_total.groupby("city")["metrics.temperature.avg"]
    df_total["kpi.temp_zscore"] = (
        df_total["metrics.temperature.avg"] - city_temperature.transform("mean")
    ) / city_temperature.transform("std")

    deleted_records = initial_records - len(df_total)
    report_path = Path(__file__).with_name("informe_limpieza.txt")
    report_path.write_text(
        f"Registros iniciales: {initial_records}\n"
        f"Registros borrados: {deleted_records}\n"
        f"Registros finales: {len(df_total)}\n",
        encoding="utf-8",
    )

    city_summary = df_total.groupby("city", as_index=False).agg(
        historical_mean_temperature=("metrics.temperature.avg", "mean"),
        total_rainfall=("metrics.precipitation.avg", "sum"),
    )
    annual_summary = df_total.groupby(
        ["city", "year"], as_index=False
    ).agg(
        absolute_max_temperature=("metrics.temperature.max", "max"),
        absolute_min_temperature=("metrics.temperature.min", "min"),
    )

    # Crea la carpeta de salida y guarda los tres DataFrames en formato Parquet con PyArrow.
    data_directory = Path(__file__).with_name("data")
    data_directory.mkdir(parents=True, exist_ok=True)
    df_total.to_parquet(
        data_directory / "df_total.parquet", engine="pyarrow", index=False
    )
    city_summary.to_parquet(
        data_directory / "city_summary.parquet", engine="pyarrow", index=False
    )
    annual_summary.to_parquet(
        data_directory / "annual_summary.parquet", engine="pyarrow", index=False
    )

    print(df_total.head())
    print("\nResumen por ciudad:")
    print(city_summary)
    print("\nResumen anual:")
    print(annual_summary)

except requests.exceptions.RequestException as e:
    print(f'Error en la petición: {e}')
