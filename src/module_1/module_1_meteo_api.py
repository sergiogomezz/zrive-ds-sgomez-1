import requests
import time
import logging
import json
import pandas as pd
from urllib.parse import urlencode
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, List

# configuraciones del logger
logger = logging.getLogger(__name__)
logger.level = logging.INFO


# DECLARO VARIABLES DE ENTORNO E INFORMACION
API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"


# 1. IMPLEMENTACION DE LA LLAMADA A LA API Y LA CAPTURA DE LOS DATOS DE METEO


# FUNCION PARA COGER LOS DATOS DE LA API
def get_data_meteo_api(
    longitude: float, latitude: float, start_date: str, end_date: str
):
    headers = {}  # vacio porque no se necesitan headers

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
    }

    return request_with_cooloff(API_URL + urlencode(params, safe=","), headers)


# FUNCION QUE LLAMA A APIs GENERICA - se utiliza cooloff dinamico en lugar de un
# time.sleep(x)
def _request_with_cooloff(
    url: str,
    headers: Dict[str, any],
    num_attemps: int,
    payload: Optional[Dict[str, any]] = None,
) -> requests.Response:
    cooloff = 1
    for call_count in range(num_attemps):
        try:
            if payload is None:
                res = requests.get(url, headers=headers)
            else:
                res = requests.get(url, headers=headers, json=payload)
            res.raise_for_status()

        except (
            requests.exceptions.ConnectionError
        ) as e:  # maneja los errores de conexion a la API
            logger.info("API refused the connection")
            logger.warning(e)
            if call_count != num_attemps - 1:
                time.sleep(cooloff)
                cooloff *= 2
                continue
            else:
                raise

        except (
            requests.exceptions.HTTPError
        ) as e:  # maneja errores comunes de la API y define los logs
            logger.warning(e)
            if res.status_code == 404:
                raise

            logger.info(f"API return code {res.status_code} cooloff at {cooloff}")
            if call_count != num_attemps - 1:
                time.sleep(cooloff)
                cooloff *= 2
                continue
            else:
                raise

        return res


# se utiliza funcion publica para exponer menor logica
def request_with_cooloff(
    url: str, headers: Dict[str, any], payload: Dict[str, any] = None, num_attemps=10
) -> Dict[any, any]:
    return json.loads(
        _request_with_cooloff(url, headers, num_attemps, payload).content.decode(
            "utf-8"
        )
    )


# 2-3. PROCESAR LOS DATOS Y PLOTTEARLOS

def compute_monthly_statistics(data: pd.DataFrame, meteo_variables: List[str]):
    data["time"] = pd.to_datetime(data["time"])

    grouped = data.groupby([data["city"], data["time"].dt.to_period("M")])

    results = []

    for (city, month), group in grouped:
        monthly_stats = {"city": city, "month": month.to_timestamp()}

        for variable in meteo_variables:
            monthly_stats[f"{variable}_max"] = group[variable].max()
            monthly_stats[f"{variable}_mean"] = group[variable].mean()
            monthly_stats[f"{variable}_min"] = group[variable].min()
            monthly_stats[f"{variable}_std"] = group[variable].std()

        results.append(monthly_stats)

    return pd.DataFrame(results)


def plot_timeseries(data: pd.DataFrame):
    rows = len(VARIABLES)
    cols = len(data["city"].unique())
    fig, axs = plt.subplots(rows, cols, figsize=(10, 6 * rows))

    # Si hay solo una fila o una columna, axs puede ser 1D, debemos asegurarnos de que sea un array 2D
    if rows == 1:
        axs = [axs]  # Si hay solo una fila, convierte axs en una lista
    if cols == 1:
        axs = [[ax] for ax in axs]  # Si hay solo una columna, haz de cada fila una lista

    for i, variable in enumerate(VARIABLES):
        for k, city in enumerate(data["city"].unique()):
            city_data = data[data["city"] == city]
            axs[i][k].plot(
                city_data["month"],
                city_data[f"{variable}_mean"],
                label=f"{city} (mean)",
                color=f"C{k}",
            )

            axs[i][k].fill_between(
                city_data["month"],
                city_data[f"{variable}_min"],
                city_data[f"{variable}_max"],
                alpha=0.2,
                color=f"C{k}",
            )

            axs[i][k].set_xlabel("Date")
            axs[i][k].set_title(variable)
            if k == 0:
                axs[i][k].set_ylabel("Value")
            axs[i][k].legend()

    plt.tight_layout()
    plt.show()


def main():
    data_list = []
    time_spam = (
        pd.date_range(START_DATE, END_DATE, freq="D").strftime("%Y-%m-%d").tolist()
    )

    for city, coord in COORDINATES.items():
        latitude = coord["latitude"]
        longitude = coord["longitude"]
        data = pd.DataFrame(
            get_data_meteo_api(longitude, latitude, START_DATE, END_DATE)["daily"]
        ).assign(city=city)
        data_list.append(data)

        data = pd.concat(data_list)

        calculates_ts = compute_monthly_statistics(data, VARIABLES)

        plot_timeseries(calculates_ts)


if __name__ == "__main__":
    main()
