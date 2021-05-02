import pandas as pd
import streamlit as st
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import diverging


HOST_URL = "https://covid.ourworldindata.org/data/"
MAIN_DATA_NAME = "owid-covid-data.csv"
TESTING_DATA_NAME = "testing/covid-testing-all-observations.csv"
VACCINATIONS_DATA_NAME = "vaccinations/vaccinations.csv"
MIN_DATE = date(2020, 1, 1)
MAX_DATE = date.today()
CASES_DEATHS_COLORS = ["red", "lightskyblue", "orange"]
VACCINATIONS_COLORS = ["greenyellow", "lightskyblue", "green"]


@st.cache(ttl=12 * 60 * 60)
def load_covid_data():
    df_main = pd.read_csv(f"{HOST_URL}/{MAIN_DATA_NAME}", parse_dates=["date"])
    df_tests = pd.read_csv(
        f"{HOST_URL}/{TESTING_DATA_NAME}", parse_dates=["Date"]
    ).rename(
        columns={
            "ISO code": "iso_code",
            "Date": "date",
            "7-day smoothed daily change": "new_tests",
        }
    )[
        ["date", "iso_code", "new_tests"]
    ]
    df_vaccinations = pd.read_csv(
        f"{HOST_URL}/{VACCINATIONS_DATA_NAME}", parse_dates=["date"]
    )[["date", "iso_code", "people_vaccinated", "people_fully_vaccinated"]]
    merge_params = dict(on=["date", "iso_code"], how="left", suffixes=("", "_merged"))
    df = df_main.merge(df_tests, **merge_params).merge(df_vaccinations, **merge_params)
    columns_to_fill = ["new_tests", "people_vaccinated", "people_fully_vaccinated"]
    for c in columns_to_fill:
        df.loc[:, c] = df[c].fillna(df[f"{c}_merged"])
    df.drop(columns=[f"{c}_merged" for c in columns_to_fill], inplace=True)
    df.drop_duplicates(subset=["date", "location"], inplace=True)
    return df


def make_title():
    st.title("Country COVID Profile")
    description_markdown_body = "\n".join(
        [
            "- Tests",
            "- New cases",
            "- New deaths",
            "- Vaccinations",
            "- Positive Test & Case Fatality Rate",
            "- Country's Strignency Index",
            "\n",
            "Use data by [Our World In Data](https://ourworldindata.org/coronavirus)",
        ]
    )
    st.markdown(description_markdown_body)


def get_locations(df, orderby="population", ascending=False):
    return (
        df.query("continent.notna()")[["location", orderby]]
        .drop_duplicates(subset="location")
        .sort_values(by=orderby, ascending=ascending)["location"]
        .reset_index(drop=True)
    )


def select_location(locations):
    location = st.sidebar.selectbox(
        "Country", locations, index=8, help="For showing COVID profile"
    )
    return location


def select_rolling_window():
    return st.sidebar.slider(
        "Rolling Window", 1, 30, 7, 1, help="Window for calculating rolling mean"
    )


def select_y_scale():
    return st.sidebar.radio(
        "Y scale",
        ["Linear", "Logarithmic"],
        index=0,
        help="Scaling Y axis for total values",
    )


def select_date_range():
    min_date = st.sidebar.date_input(
        "Start Date",
        MAX_DATE - timedelta(days=90),
        MIN_DATE,
        MAX_DATE,
        help="First day in shown data",
    )
    max_date = st.sidebar.date_input(
        "End Date", MAX_DATE, MIN_DATE, MAX_DATE, help="Last day in shown data"
    )
    return min_date, max_date


def filter_data(input_df, location, min_date, max_date):
    condition = f"location == '{location}' and '{min_date}' <= date <= '{max_date}'"
    return input_df.query(condition)


def make_legend_name(column_name):
    capitalized_words = [w.capitalize() for w in column_name.split("_")]
    return " ".join(capitalized_words)


def make_total_and_rate_plot(df, numerator, denominator, rate, y_scale, colors):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df.index
    numerator_name = make_legend_name(numerator)
    denominator_name = make_legend_name(denominator)
    rate_name = make_legend_name(rate)
    common_plot_params = dict(mode="lines+markers", line_shape="spline")
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[numerator],
            name=numerator_name,
            fill="tozeroy",
            line=dict(color=colors[0]),
            hovertemplate="%{y:,.0f}",
            **common_plot_params,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[denominator],
            name=denominator_name,
            fill="tonexty",
            line=dict(color=colors[1]),
            hovertemplate="%{y:,.0f}",
            **common_plot_params,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[rate],
            name=rate_name,
            line=dict(color=colors[2]),
            hovertemplate="%{y:.1%}",
            **common_plot_params,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title_text=f"{numerator_name} & {denominator_name}", hovermode="x"
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(
        title_text="Total",
        secondary_y=False,
        type="linear" if y_scale == "Linear" else "log",
    )
    fig.update_yaxes(title_text="Rate", tickformat="%", secondary_y=True)
    st.plotly_chart(fig)


def choose_colors_for_plot(numerator):
    if numerator in ["new_cases", "new_deaths"]:
        return CASES_DEATHS_COLORS
    elif numerator in ["people_vaccinated", "people_fully_vaccinated"]:
        return VACCINATIONS_COLORS


def plot_covid_stat(df, numerator, denumerator, rate, rolling_window, y_scale, colors):
    plot_data = df.set_index("date")[[numerator, denumerator]]
    plot_data = plot_data.rolling(rolling_window).mean().dropna()
    plot_data.loc[:, rate] = plot_data[numerator] / plot_data[denumerator]
    make_total_and_rate_plot(plot_data, numerator, denumerator, rate, y_scale, colors)


def plot_stringency_index(df):
    plot_data = df[["date", "stringency_index"]]
    fig = px.bar(
        df,
        x="date",
        y="stringency_index",
        color="stringency_index",
        color_continuous_scale=diverging.Geyser,
        range_color=[0, 100],
        color_continuous_midpoint=50,
    )
    fig.update_layout(title_text="Measure of the strictness of policy responses")
    fig.update_yaxes(title_text="Stringency Index", range=[0, 100])
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig)


def main():
    make_title()
    df_all = load_covid_data()
    locations = get_locations(df_all)
    location = select_location(locations)
    min_date, max_date = select_date_range()
    df = filter_data(df_all, location, min_date, max_date)
    rolling_window = select_rolling_window()
    y_scale = select_y_scale()
    for numerator, denumerator, rate in [
        ["new_cases", "new_tests", "positive_test_rate"],
        ["new_deaths", "new_cases", "case_fatality_rate"],
        ["people_vaccinated", "population", "vaccination_rate"],
        ["people_fully_vaccinated", "population", "fully_vaccination_rate"],
    ]:
        colors = choose_colors_for_plot(numerator)
        plot_covid_stat(
            df, numerator, denumerator, rate, rolling_window, y_scale, colors
        )
    plot_stringency_index(df)


if __name__ == "__main__":
    main()
