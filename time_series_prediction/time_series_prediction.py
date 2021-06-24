from datetime import date, timedelta
from generate_time_series import (
    generate_time_series,
    WEEK_VALUES_INDEX_INFO,
    MONTH_VALUES_INDEX_INFO,
    scale_date,
)
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from fbprophet import Prophet


def make_title():
    st.title("Time Series Forecasting")
    description_markdown_body = "\n\n".join(
        [
            ":hammer_and_wrench: Create your own dataset",
            ":watch: Choose train and validation periods",
            ":crystal_ball: Train models on train data",
            ":mag_right: Check forecasts on validation data",
            ":hourglass: ...",
            ":tada: PROFIT!",
        ]
    )
    st.markdown(description_markdown_body)


def make_header(text, sidebar=False):
    if sidebar:
        st.sidebar.header(text)
    else:
        st.header(text)


def choose_date(date_name, initial_date, min_date, max_date, sidebar=False):
    widget_params = dict(
        label=date_name, value=initial_date, min_value=min_date, max_value=max_date
    )
    if sidebar:
        return st.sidebar.date_input(**widget_params)
    else:
        return st.date_input(**widget_params)


def normalize_seasonality_coefficients(seasons_info):
    sum_values = sum(seasons_info.values())
    return {
        key: value - sum_values / len(seasons_info)
        for key, value in seasons_info.items()
    }


def choose_seasonality(season_name):
    if season_name == "week":
        season_options = WEEK_VALUES_INDEX_INFO.keys()
    elif season_name == "month":
        season_options = MONTH_VALUES_INDEX_INFO.keys()
    high_season_name = st.sidebar.selectbox(
        f"Choose {season_name}ly seasonality coefficients",
        ["NO"] + list(season_options),
        help="No seasonality OR Choose HIGH season",
    )
    if high_season_name != "NO":
        season_factor = st.sidebar.slider(
            f"Sesonality of {season_name} factor",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            help="Higher value = stronger season effect",
        )
        return [high_season_name, season_factor]


def create_time_series():
    make_header("Create your time series", True)
    today = date.today()
    start = choose_date("Start Date", today - timedelta(360), None, None, True)
    end = choose_date("End Date", today + timedelta(30), start, None, True)
    scale = st.sidebar.selectbox(
        "Date Scale", ["D", "W", "M"], help="D = day, W = week, M = month"
    )
    initial_value = st.sidebar.number_input(
        "Initial Value in Series", value=100, help="Trend begins with this number"
    )
    trend_coeff = st.sidebar.slider(
        "Trend Line Coefficient",
        min_value=-1.0,
        max_value=1.0,
        value=0.1,
        help="Series change by rate X in the end",
    )
    monthly_seasonality = choose_seasonality("month")
    weekly_seasonality = choose_seasonality("week")
    max_random_effect = st.sidebar.slider(
        "Max Random Effect",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        help="Higher value = more randomness",
    )
    series = generate_time_series(
        start,
        end,
        scale,
        initial_value,
        trend_coeff,
        monthly_seasonality,
        weekly_seasonality,
        max_random_effect,
    )
    return series, scale


def make_train_and_test(series, test_proportion=0.2):
    make_header("Define Train and Test Data")
    st.markdown("● Train: date < Split Date")
    st.markdown("● Test: date >= Split Date")
    split_date = pd.to_datetime(
        choose_date(
            "Split Date",
            series.index[int((1 - test_proportion) * len(series))],
            series.index.min(),
            series.index.max(),
        )
    )
    train = series[series.index < split_date]
    test = series[series.index >= split_date]
    return train, test, split_date


def plot_series(series, split_date):
    if len(series) > 0:
        fig = px.line(series)
        fig.add_vline(x=split_date, line=dict(color="red", dash="dash"))
        fig.update_traces(mode="markers+lines", hovertemplate=None)
        fig.update_yaxes(title_text="Value")
        fig.update_layout(title_text="Your Series", showlegend=False)
        st.plotly_chart(fig)


def train_fbprophet(train, scale):
    data_for_model = train.reset_index().rename(columns={"date": "ds", "value": "y"})
    if scale == "D":
        model_params = dict(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False
        )
    elif scale == "W":
        model_params = dict(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
        )
    elif scale == "M":
        model_params = dict(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
    return Prophet(**model_params).fit(data_for_model)


def apply_fbprophet(model, test):
    suffixes = ["_lower", "", "_upper"]
    data_for_model = pd.DataFrame(data=test.index.values, columns=["ds"])
    return (
        model.predict(data_for_model)[["ds"] + [f"yhat{s}" for s in suffixes]]
        .rename(columns={"ds": "date"})
        .rename(columns={f"yhat{s}": f"forecast{s}" for s in suffixes})
        .set_index("date")
    )


def make_forecast_with_frophet(train, test, scale):
    model = train_fbprophet(train, scale)
    forecast = apply_fbprophet(model, test)
    forecast.loc[:, "real"] = test.values
    return forecast


def plot_data_and_forecast(train, forecast, model_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.forecast_lower,
            name="Lower Boundary",
            line_color="red",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.forecast_upper,
            name="Upper Boundary",
            line_color="red",
            fill="tonexty",
        )
    )
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.real, name="Test Series"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.forecast, name="Forecast"))
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train Series"))
    fig.update_xaxes(title_text="Date")
    fig.update_layout(title_text=model_name, hovermode="x")
    st.plotly_chart(fig)


if __name__ == "__main__":
    make_title()
    series, scale = create_time_series()
    train, test, split_date = make_train_and_test(series)
    plot_series(series, split_date)
    fb_forecast = make_forecast_with_frophet(train, test, scale)
    plot_data_and_forecast(train, fb_forecast, "Facebook Prophet")
