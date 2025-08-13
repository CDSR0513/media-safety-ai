import pandas as pd
import plotly.express as px

def daily_bar(df: pd.DataFrame):
    g = df.groupby("date")["total"].sum().reset_index()
    return px.bar(g, x="date", y="total")

def platform_bar(df: pd.DataFrame):
    g = df.groupby("platform")["total"].sum().reset_index()
    return px.bar(g, x="platform", y="total")
