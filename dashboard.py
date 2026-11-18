from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


DATA_FILE = Path(__file__).parent / "data" / "df_total.parquet"
CITIES = [
    ("MAD", "Madrid", "#d62728", "rgba(214, 39, 40, 0.16)"),
    ("LON", "Londres", "#1f77b4", "rgba(31, 119, 180, 0.16)"),
]
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
          "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def city_label(city_code):
    return next(name for code, name, _, _ in CITIES if code == city_code)


st.set_page_config(page_title="Clima Madrid y Londres", layout="wide")
st.title("Comparativa climática: Madrid y Londres")

if not DATA_FILE.exists():
    st.error("No se encuentra data/df_total.parquet. Ejecuta primero app.py.")
    st.stop()

df_total = pd.read_parquet(DATA_FILE, engine="pyarrow")
required_columns = {
    "city",
    "year",
    "month",
    "metrics.temperature.max",
    "metrics.temperature.avg",
    "metrics.temperature.min",
    "metrics.precipitation.avg",
    "thermal_range",
}
missing_columns = required_columns.difference(df_total.columns)
if missing_columns:
    st.error(f"Faltan columnas requeridas en el Parquet: {sorted(missing_columns)}")
    st.stop()

df_total["date"] = pd.to_datetime(
    df_total[["year", "month"]].assign(day=1)
)

# Muestra primero el último dato disponible de cada ciudad.
st.subheader("Resumen instantáneo")
summary_columns = st.columns(len(CITIES))
for column, (city_code, city_name, color, _) in zip(summary_columns, CITIES):
    city_data = df_total[df_total["city"] == city_code].sort_values("date")
    with column:
        st.markdown(
            f"<h3 style='color: {color}'>{city_name}</h3>",
            unsafe_allow_html=True,
        )
        if city_data.empty:
            st.info("No hay datos disponibles para esta ciudad.")
            continue

        latest = city_data.iloc[-1]
        month_name = MONTHS[int(latest["month"]) - 1]
        st.caption(f"Último dato: {month_name} de {int(latest['year'])}")
        first_row, second_row = st.columns(2)
        first_row.metric(
            "Temperatura media",
            f"{latest['metrics.temperature.avg']:.1f} °C",
        )
        second_row.metric(
            "Precipitación",
            f"{latest['metrics.precipitation.avg']:.1f} mm",
        )
        first_row.metric(
            "Temperatura máxima",
            f"{latest['metrics.temperature.max']:.1f} °C",
        )
        second_row.metric(
            "Temperatura mínima",
            f"{latest['metrics.temperature.min']:.1f} °C",
        )

# La banda coloreada representa el intervalo entre la mínima y la máxima.
st.header("Evolución de la temperatura")
evolution_figure = go.Figure()
for city_code, city_name, color, fill_color in CITIES:
    city_data = df_total[df_total["city"] == city_code].sort_values("date")
    if city_data.empty:
        continue

    evolution_figure.add_trace(
        go.Scatter(
            x=city_data["date"],
            y=city_data["metrics.temperature.min"],
            mode="lines",
            line={"width": 0, "color": color},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    evolution_figure.add_trace(
        go.Scatter(
            x=city_data["date"],
            y=city_data["metrics.temperature.max"],
            mode="lines",
            line={"width": 0, "color": color},
            fill="tonexty",
            fillcolor=fill_color,
            name=f"{city_name}: rango mín.-máx.",
            hovertemplate="%{x|%b %Y}<br>Máxima: %{y:.1f} °C<extra></extra>",
        )
    )
    evolution_figure.add_trace(
        go.Scatter(
            x=city_data["date"],
            y=city_data["metrics.temperature.avg"],
            mode="lines",
            line={"color": color, "width": 2},
            name=f"{city_name}: media",
            hovertemplate="%{x|%b %Y}<br>Media: %{y:.1f} °C<extra></extra>",
        )
    )

evolution_figure.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Mes",
    yaxis_title="Temperatura (°C)",
    legend_title="Ciudad",
    hovermode="x unified",
)
st.plotly_chart(evolution_figure, width="stretch")

historical_temperature = df_total.groupby("city")["metrics.temperature.avg"].mean()
average_range = df_total.groupby("city")["thermal_range"].mean()
warmest_city = historical_temperature.idxmax()
widest_range_city = average_range.idxmax()
st.markdown(
    f"**Interpretación:** En el periodo disponible, {city_label(warmest_city)} "
    f"tiene la mayor temperatura media histórica "
    f"({historical_temperature[warmest_city]:.1f} °C). "
    f"{city_label(widest_range_city)} presenta la mayor oscilación mensual "
    f"media entre máximas y mínimas ({average_range[widest_range_city]:.1f} °C)."
)

# Compara las medias históricas de cada mes; se separan las métricas por escala.
st.header("Perfil climático mensual")
monthly_profile = df_total.groupby(["city", "month"], as_index=False).agg(
    mean_temperature=("metrics.temperature.avg", "mean"),
    mean_precipitation=("metrics.precipitation.avg", "mean"),
)
radar_figure = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "polar"}, {"type": "polar"}]],
    subplot_titles=("Temperatura media (°C)", "Precipitación media (mm)"),
)

for column_index, metric, unit in [
    (1, "mean_temperature", "°C"),
    (2, "mean_precipitation", "mm"),
]:
    for city_code, city_name, color, fill_color in CITIES:
        city_profile = (
            monthly_profile[monthly_profile["city"] == city_code]
            .set_index("month")
            .reindex(range(1, 13))
        )
        radar_figure.add_trace(
            go.Scatterpolar(
                theta=MONTHS,
                r=city_profile[metric],
                mode="lines+markers",
                fill="toself",
                fillcolor=fill_color,
                line={"color": color, "width": 2},
                marker={"color": color, "size": 6},
                name=city_name,
                showlegend=column_index == 1,
                hovertemplate=f"%{{theta}}<br>%{{r:.1f}} {unit}<extra>{city_name}</extra>",
            ),
            row=1,
            col=column_index,
        )

radar_figure.update_layout(
    template="plotly_white",
    height=520,
    showlegend=True,
    legend_title="Ciudad",
    polar={"angularaxis": {"categoryorder": "array", "categoryarray": MONTHS}},
    polar2={"angularaxis": {"categoryorder": "array", "categoryarray": MONTHS}},
)
st.plotly_chart(radar_figure, width="stretch")

# Resume los meses más cálidos y lluviosos para hacer explícita la comparación.
for city_code, city_name, _, _ in CITIES:
    city_profile = monthly_profile[monthly_profile["city"] == city_code]
    if city_profile.empty:
        continue

    warmest_month = city_profile.loc[city_profile["mean_temperature"].idxmax()]
    wettest_month = city_profile.loc[city_profile["mean_precipitation"].idxmax()]
    st.markdown(
        f"**{city_name}:** el mes más cálido en promedio es "
        f"{MONTHS[int(warmest_month['month']) - 1]} "
        f"({warmest_month['mean_temperature']:.1f} °C); el más lluvioso es "
        f"{MONTHS[int(wettest_month['month']) - 1]} "
        f"({wettest_month['mean_precipitation']:.1f} mm)."
    )

st.caption(
    "Los radares muestran promedios históricos por mes. "
    "Temperatura y precipitación se presentan en paneles separados para "
    "compararlas sin mezclar sus escalas."
)
