"""稳健性：债券腿换三种口径。运行：uv run python -m macrob.robust_bond

主方案是中债国债财富指数（7–10 年）；对照是十年国债 ETF（511260，2017 年 8 月上市，后复权）和用收益率合成的 10 年国债。
其余规则和 allocation 完全一样。ETF 要攒够 52 周数据才能估协方差，所以区间从 2018 年下半年开始。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import allocation as A
from .datalayer import RESULTS
from .probe_util import RAW


def etf_levels() -> pd.DataFrame:
    etf = pd.read_csv(RAW / "etf_511260_hfq.csv", parse_dates=["日期"]).set_index("日期")["收盘"].astype(float)
    lv = A.daily_levels()
    lv["债"] = etf.reindex(lv.index)
    return lv


def main() -> None:
    cash = A.cash_monthly()
    q = pd.read_csv(RESULTS / "quadrant_probabilities.csv", index_col=0, parse_dates=True)
    q.index = q.index + pd.offsets.MonthEnd(0)
    prob = lambda t: np.array([q.loc[t, "增长加速"] * (1 - q.loc[t, "通胀加速"]), q.loc[t, "增长加速"] * q.loc[t, "通胀加速"], 1 - q.loc[t, "增长加速"]])
    def fixed(t):
        g, i = q.loc[:t, "增长加速"], q.loc[:t, "通胀加速"]
        return np.array([(g * (1 - i)).mean(), (g * i).mean(), (1 - g).mean()])
    rows = []
    synth = A.daily_levels().assign(债=lambda d: d["债(合成)"])
    for leg, lv in (("中债国债财富 7–10 年", A.daily_levels()), ("十年国债 ETF", etf_levels()), ("合成 10 年国债", synth)):
        months = pd.date_range(q.index.min(), min(q.index.max(), lv.index.max()), freq="ME")
        res = {"60/40（股 / 债）": A.run(None, lv, months, fixed_w=[0.6, 0.0, 0.4]),
               "风险平价": A.run(lambda t: np.ones(3) / 3, lv, months, max_to=None),
               "固定预算（不择时）": A.run(fixed, lv, months, max_to=None),
               "概率预算（择时）": A.run(prob, lv, months)}
        rows.append((leg, res))
    start = max(r.index.min() for _, res in rows for r, _ in res.values())
    out = []
    for leg, res in rows:
        for name, (r, W) in res.items():
            r = r[r.index >= start]
            m = A.metrics(r, cash)
            out.append({"债券腿": leg, "方案": name, "起": f"{r.index.min():%Y-%m}", **{k: m[k] for k in ("月数", "年化", "波动", "扣无风险夏普", "最大回撤")}})
    t = pd.DataFrame(out)
    t.to_csv(RESULTS / "allocation_bond_etf.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
