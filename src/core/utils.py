
import pandas as pd
import numpy as np

def cagr(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1

def mdd(series):
    roll_max = series.cummax()
    drawdown = series / roll_max - 1
    return drawdown.min()
