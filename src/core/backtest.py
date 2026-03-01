import yfinance as yf
import pandas as pd
import numpy as np
from .regime import compute_regime


def download(cfg):
    df = yf.download(
        cfg["data"]["tickers"],
        start=cfg["data"]["start"],
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("Downloaded price data is empty.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.levels[0]:
            df = df["Adj Close"]
        else:
            df = df["Close"]
    else:
        if "Adj Close" in df.columns:
            df = df[["Adj Close"]]
        else:
            df = df[["Close"]]

    df = df.dropna()

    if df.empty:
        raise ValueError("Price data empty after cleaning.")

    return df


def momentum_rank(df):
    ret = df.pct_change(126)
    return ret.rank(axis=1, ascending=False)


def apply_risk_off(mode):
    if mode == "SHY_100":
        return {"SHY": 1.0}
    if mode == "SHY_GLD_50_50":
        return {"SHY": 0.5, "GLD": 0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY": 0.7, "GLD": 0.3}
    if mode == "GLD_100":
        return {"GLD": 1.0}
    return {}


def run(cfg):
    prices = download(cfg)

    returns = prices.pct_change().fillna(0)

    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    ranks = momentum_rank(prices)
    regime = compute_regime(prices.iloc[:, 0], cfg)

    equity = 1.0
    curve = []

    for date in prices.index:
        state = regime.loc[date]

        if state == "RISK_OFF":
            weights = apply_risk_off(cfg["risk_off"]["mode"])
            daily_ret = 0.0
            for t, w in weights.items():
                if t in returns.columns:
                    daily_ret += returns.loc[date, t] * w
        else:
            top = ranks.loc[date].nsmallest(cfg["selection"]["top_n"]).index
            lev = (
                cfg["leverage"]["strong"]
                if state == "STRONG"
                else cfg["leverage"]["weak"]
            )
            daily_ret = returns.loc[date, top].mean() * lev

        equity *= (1 + (daily_ret if pd.notna(daily_ret) else 0))
        curve.append(equity)

    if len(curve) == 0:
        raise ValueError("Equity curve is empty.")

    return pd.Series(curve, index=prices.index)