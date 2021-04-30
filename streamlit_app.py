COVID_DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
import pandas as pd

def make_simple_plot(df, location, metric):
    df.query("location == @location").plot(x="date", y=metric, grid=True, title=f"{location}: {metric}")

if __name__ == "__main__":
    df = pd.read_csv(COVID_DATA_URL, parse_dates=["date"])
    print("Hello, world")
    make_simple_plot(df, location="Russia", metric="new_cases")
