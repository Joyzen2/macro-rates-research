"""拐点捕捉：每个参考拐点，在前后 6 个月内找同方向、还没被配对过的模型切换，取最近的一个。

- 捕捉率 = 配上的参考拐点 ÷ 全部参考拐点。
- 提前月数 = 参考拐点月份 − 模型切换月份（正数表示模型更早）。
- 误报 = 没有配上任何参考拐点的模型切换。
只评估两者都有数据的时段。
"""
from __future__ import annotations

import pandas as pd

WINDOW = 6


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def score(ref: pd.DataFrame, sig: pd.DataFrame, start, end) -> dict:
    ref = ref[(ref["date"] >= start) & (ref["date"] <= end)].reset_index(drop=True)
    sig = sig[(sig["date"] >= start - pd.DateOffset(months=WINDOW)) & (sig["date"] <= end + pd.DateOffset(months=WINDOW))].reset_index(drop=True)
    used, leads, detail = set(), [], []
    for _, r in ref.iterrows():
        cands = [(abs(_months(r["date"], s["date"])), j) for j, s in sig.iterrows()
                 if s["kind"] == r["kind"] and j not in used and abs(_months(r["date"], s["date"])) <= WINDOW]
        if cands:
            _, j = min(cands)
            used.add(j)
            leads.append(_months(r["date"], sig.loc[j, "date"]))
            detail.append((r["date"].date(), r["kind"], sig.loc[j, "date"].date(), leads[-1]))
        else:
            detail.append((r["date"].date(), r["kind"], None, None))
    n = len(ref)
    return {"参考拐点数": n, "捕捉数": len(leads), "捕捉率": len(leads) / n if n else float("nan"),
            "平均提前月数": sum(leads) / len(leads) if leads else float("nan"),
            "模型切换数": len(sig[(sig["date"] >= start) & (sig["date"] <= end)]), "误报数": len(sig) - len(used), "明细": detail}
