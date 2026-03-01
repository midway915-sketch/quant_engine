
import yaml, argparse, itertools, copy, os
from src.core.backtest import run
from src.core.utils import cagr, mdd
import pandas as pd

def set_deep(d,k,v):
    keys=k.split(".")
    cur=d
    for kk in keys[:-1]:
        cur=cur[kk]
    cur[keys[-1]]=v

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--grid",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()

    base=yaml.safe_load(open(args.config))
    grid=yaml.safe_load(open(args.grid))

    keys=list(grid.keys())
    combos=list(itertools.product(*grid.values()))

    results=[]

    for vals in combos:
        cfg=copy.deepcopy(base)
        for k,v in zip(keys,vals):
            set_deep(cfg,k,v)

        curve=run(cfg)
        results.append({
            "params":str(dict(zip(keys,vals))),
            "CAGR":cagr(curve),
            "MDD":mdd(curve)
        })

    df=pd.DataFrame(results).sort_values("CAGR",ascending=False)
    os.makedirs(args.out,exist_ok=True)
    df.to_csv(f"{args.out}/summary.csv",index=False)

if __name__=="__main__":
    main()
