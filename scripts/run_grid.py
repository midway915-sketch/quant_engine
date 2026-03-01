import yaml
import argparse
import itertools
import copy
import os
import pandas as pd

from src.core.backtest import run
from src.core.utils import cagr, mdd


def set_deep(d, k, v):
    keys = k.split(".")
    cur = d
    for kk in keys[:-1]:
        cur = cur[kk]
    cur[keys[-1]] = v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = yaml.safe_load(open(args.config))
    grid = yaml.safe_load(open(args.grid))

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))

    results = []

    print(f"[DEBUG] Total combinations: {len(combos)}")

    for i, vals in enumerate(combos):
        cfg = copy.deepcopy(base)

        for k, v in zip(keys, vals):
            set_deep(cfg, k, v)

        print(f"\n[DEBUG] Running combo {i+1}/{len(combos)}")
        print(f"[DEBUG] Params: {dict(zip(keys, vals))}")

        curve = run(cfg)

        # 🔥 디버그: 곡선 길이 확인
        if curve is None or len(curve) == 0:
            print("[ERROR] Empty equity curve.")
            continue

        start_date = curve.index[0]
        end_date = curve.index[-1]
        years = (end_date - start_date).days / 365.25

        print(f"[DEBUG] Start: {start_date}")
        print(f"[DEBUG] End: {end_date}")
        print(f"[DEBUG] Years: {years:.2f}")
        print(f"[DEBUG] Final equity: {curve.iloc[-1]}")

        # 🔥 안전장치
        if years < 5:
            print("[WARNING] Backtest period < 5 years. Possible distortion.")

        calc_cagr = cagr(curve)
        calc_mdd = mdd(curve)

        print(f"[DEBUG] CAGR: {calc_cagr}")
        print(f"[DEBUG] MDD: {calc_mdd}")

        results.append({
            "params": str(dict(zip(keys, vals))),
            "CAGR": calc_cagr,
            "MDD": calc_mdd,
            "start": start_date,
            "end": end_date,
            "years": years,
            "final_equity": curve.iloc[-1]
        })

    df = pd.DataFrame(results).sort_values("CAGR", ascending=False)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(f"{args.out}/summary.csv", index=False)

    print("\n[DEBUG] Grid run complete.")
    print(df.head())


if __name__ == "__main__":
    main()