from numpy.random import uniform
import pandas as pd
from datetime import date, timedelta
from typing import Optional


WEEK_VALUES_INDEX_INFO = {
    "MON-FRI": [i for i in range(5)],
    "SAT-SUN": [5, 6],
}
MONTH_VALUES_INDEX_INFO = {
    "SUMMER": [6, 7, 8],
    "WINTER": [12, 1, 2],
}


def create_seasonality_dict(
    season_factor: float,
    high_season_indexes: list,
    low_season_indexes: list,
) -> dict:
    result = {}
    min_index = min(min(high_season_indexes), min(low_season_indexes))
    max_index = max(max(high_season_indexes), max(low_season_indexes))
    for i in range(min_index, max_index + 1):
        if i in high_season_indexes:
            result[i] = season_factor
        elif i in low_season_indexes:
            result[i] = -season_factor
        else:
            result[i] = 0
    return result


def define_seasonality(df, user_input, season_name):
    if user_input is None:
        seasonality_effect = 0
        df.loc[:, f"{season_name}_effect"] = 0
    else:
        if season_name == "week":
            values_index_info = WEEK_VALUES_INDEX_INFO
            datetime_func = lambda t: t.dayofweek
        elif season_name == "month":
            values_index_info = MONTH_VALUES_INDEX_INFO
            datetime_func = lambda t: t.month
        high_season_name, high_season_value = user_input
        low_season_name = [
            key for key in values_index_info.keys() if key != high_season_name
        ][0]
        seasonality_info = create_seasonality_dict(
            high_season_value,
            values_index_info[high_season_name],
            values_index_info[low_season_name],
        )
        seasonality_effect = (
            df["date"].apply(datetime_func).apply(lambda x: seasonality_info[x])
        )
    df.loc[:, f"{season_name}_effect"] = seasonality_effect


def scale_date(d, scale):
    return pd.to_datetime(d).to_period(scale).to_timestamp()


def generate_time_series(
    start: date,
    end: date,
    scale: str,
    initial_value: float,
    trend_coeff: float,
    monthly_seasonality: Optional[list],
    weekly_seasonality: Optional[list],
    max_random_effect: float,
) -> pd.Series:
    start_scaled = scale_date(start, scale)
    end_scaled = scale_date(end, scale) - timedelta(1)
    df = pd.DataFrame(pd.date_range(start_scaled, end_scaled, name="date"))
    df.loc[:, "trend"] = trend_coeff / (end_scaled - start_scaled).days * df.index
    define_seasonality(df, monthly_seasonality, "month")
    define_seasonality(df, weekly_seasonality, "week")
    df.loc[:, "random_effect"] = [
        uniform(-max_random_effect, max_random_effect) for _ in range(len(df))
    ]
    df.loc[:, "value"] = initial_value * (
        1 + df["trend"] + df["week_effect"] + df["month_effect"] + df["random_effect"]
    )
    df.loc[:, "date"] = df["date"].apply(lambda d: scale_date(d, scale))
    series = df.groupby("date")["value"].sum()
    if scale == "D":
        normalize_coeff = 1
    elif scale == "W":
        normalize_coeff = 7
    elif scale == "M":
        normalize_coeff = 30
    return series / normalize_coeff


# print(
#     generate_time_series(
#         date(2021, 1, 1),
#         date(2021, 2, 1),
#         "D",
#         100,
#         0.5,
#         ["SUMMER", 0],
#         ["MON-FRI", 0],
#         0,
#     )
# )
