import pandas as pd
import streamlit as st
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import diverging


COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
# METRICS = [
#     "new_tests",
#     "new_tests_per_thousand",
#     "new_cases",
#     "new_cases_per_million",
#     "new_deaths",
#     "new_deaths_per_million",
#     "new_vaccinations",
#     "total_vaccinations",
#     "total_vaccinations_per_hundred",
#     "people_vaccinated",
#     "people_vaccinated_per_hundred",
#     "people_fully_vaccinated",
#     "people_fully_vaccinated_per_hundred",
#     "stringency_index",
# ]
MIN_DATE = date(2020, 1, 1)
MAX_DATE = date.today()


@st.cache(ttl=12 * 60 * 60)
def load_covid_data():
    df = pd.read_csv(COVID_DATA_URL, parse_dates=["date"])
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
        "Country", locations, index=8, help="For showing covid profile"
    )
    return location


def show_last_update_for_location(df, location):
    last_update = df.query("location == @location")["date"].max().date()
    st.write(f"Last update for {location}: {last_update}")


def select_rolling_window():
    return st.sidebar.slider(
        "Rolling Window", 1, 30, 7, 1, help="Window for calcilating rolling mean"
    )


def select_y_scale():
    return st.sidebar.selectbox(
        "Y scale", ["Linear", "Logarithmic"], index=0, help="Scalling Y axis"
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


def make_total_and_rate_plot(df, numerator, denominator, rate, y_scale):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df.index
    numerator_name = make_legend_name(numerator)
    denominator_name = make_legend_name(denominator)
    rate_name = make_legend_name(rate)
    mode = "lines+markers"
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[numerator],
            name=numerator_name,
            fill="tozeroy",
            mode=mode,
            hovertemplate="%{y:,.0f}",
            connectgaps=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[denominator],
            name=denominator_name,
            fill="tonexty",
            mode=mode,
            hovertemplate="%{y:,.0f}",
            connectgaps=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[rate],
            name=rate_name,
            mode=mode,
            line=dict(color="black"),
            hovertemplate="%{y:.1%}",
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
    fig.update_yaxes(title_text="Rate", secondary_y=True)
    st.plotly_chart(fig)


def plot_covid_stat(df, numerator, denumerator, rate, rolling_window, y_scale):
    plot_data = df.set_index("date")[[numerator, denumerator]]
    plot_data.loc[:, rate] = plot_data[numerator] / plot_data[denumerator]
    plot_data = plot_data.rolling(rolling_window).mean().dropna()
    make_total_and_rate_plot(plot_data, numerator, denumerator, rate, y_scale)


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
    st.plotly_chart(fig)


def main():
    make_title()
    df_all = load_covid_data()
    locations = get_locations(df_all)
    location = select_location(locations)
    show_last_update_for_location(df_all, location)
    min_date, max_date = select_date_range()
    df = filter_data(df_all, location, min_date, max_date)
    rolling_window = select_rolling_window()
    y_scale = select_y_scale()
    for numerator, denumerator, rate in [
        ("new_cases", "new_tests", "positive_test_rate"),
        ("new_deaths", "new_cases", "case_fatality_rate"),
        ("people_vaccinated", "population", "vaccination_rate"),
        ("people_fully_vaccinated", "population", "fully_vaccination_rate"),
    ]:
        plot_covid_stat(df, numerator, denumerator, rate, rolling_window, y_scale)
    plot_stringency_index(df)


if __name__ == "__main__":
    main()
