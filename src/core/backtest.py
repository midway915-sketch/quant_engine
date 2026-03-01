
import yfinance as yf
import pandas as pd
import numpy as np
from .regime import compute_regime

def download(cfg):
    df = yf.download(cfg["data"]["tickers"], start=cfg["data"]["start"], progress=False)["Adj Close"]
    return df.dropna()

def momentum_rank(df):
    ret = df.pct_change(126)
    return ret.rank(axis=1, ascending=False)

def apply_risk_off(mode, row):
    if mode == "SHY_100":
        return {"SHY":1.0}
    if mode == "SHY_GLD_50_50":
        return {"SHY":0.5,"GLD":0.5}
    if mode == "SHY_70_GLD_30":
        return {"SHY":0.7,"GLD":0.3}
    return {}

def run(cfg):
    prices = download(cfg)
    ranks = momentum_rank(prices)
    regime = compute_regime(prices.iloc[:,0], cfg)

    equity = 1.0
    curve = []

    for date in prices.index:
        state = regime.loc[date]

        if state == "RISK_OFF":
            weights = apply_risk_off(cfg["risk_off"]["mode"], prices.loc[date])
            daily_ret = 0
            for t,w in weights.items():
                if t in prices.columns:
                    daily_ret += prices[t].pct_change().loc[date] * w
        else:
            top = ranks.loc[date].nsmallest(cfg["selection"]["top_n"]).index
            lev = cfg["leverage"]["strong"] if state=="STRONG" else cfg["leverage"]["weak"]
            daily_ret = prices[top].pct_change().loc[date].mean() * lev

        equity *= (1 + (daily_ret if pd.notna(daily_ret) else 0))
        curve.append(equity)

    return pd.Series(curve, index=prices.index)
