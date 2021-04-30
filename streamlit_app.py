COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
import streamlit as st
import pandas as pd

def make_simple_plot(df, location, metric):
    plot_df = df.query("location == @location").set_index("date")[metric]
    st.line_chart(plot_df)

if __name__ == "__main__":
    df = pd.read_csv(COVID_DATA_URL, parse_dates=["date"])
    st.write("Hello, world")
    make_simple_plot(df, location="Russia", metric="new_cases")
