
import pandas as pd

def compute_regime(price, cfg):
    ma_fast = price.rolling(cfg['regime']['ma_fast']).mean()
    ma_slow = price.rolling(cfg['regime']['ma_slow']).mean()
    vol = price.pct_change().rolling(cfg['regime']['vol_lookback']).std()

    strong = (price > ma_fast) & (vol < cfg['regime']['vol_spike'])
    risk_off = (price < ma_fast) | (price < ma_slow)

    regime = pd.Series("WEAK", index=price.index)
    regime[strong] = "STRONG"
    regime[risk_off] = "RISK_OFF"
    return regime
