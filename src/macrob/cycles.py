"""参考周期：用 Bry-Boschan 方法给参考序列定拐点（事后定的「标准答案」）。

参数事先定好：先做 3 个月居中移动平均，在前后各 5 个月的窗口里找局部极值；
峰谷必须交替；每个阶段至少 6 个月，一个完整周期至少 15 个月；序列两端 6 个月内不定拐点。
参考序列：制造业 PMI（主），规模以上工业增加值当月同比（稳健性检验）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd



WIN, MIN_PHASE, MIN_CYCLE, EDGE = 5, 6, 15, 6


def pmi() -> pd.Series:
    from .datalayer import raw_panel
    return raw_panel()["pmi"].dropna().rename("PMI")


def industrial_yoy() -> pd.Series:
    from .datalayer import raw_panel
    return raw_panel()["ip_yoy"].dropna().resample("ME").last().interpolate(limit=1).rename("工业增加值同比")


def turning_points(s: pd.Series) -> pd.DataFrame:
    x = s.rolling(3, center=True, min_periods=2).mean().dropna()
    v, idx = x.to_numpy(), x.index
    cand = []
    for i in range(EDGE, len(v) - EDGE):
        w = v[max(0, i - WIN): i + WIN + 1]
        if v[i] == w.max():
            cand.append((i, "peak"))
        elif v[i] == w.min():
            cand.append((i, "trough"))
    # 峰谷交替：连续两个同类拐点只留更极端的那个
    alt = []
    for i, k in cand:
        if alt and alt[-1][1] == k:
            j = alt[-1][0]
            better = (v[i] > v[j]) if k == "peak" else (v[i] < v[j])
            if better:
                alt[-1] = (i, k)
        else:
            alt.append((i, k))
    # 最短阶段和最短周期：不满足时去掉这一对里较弱的
    changed = True
    while changed and len(alt) > 2:
        changed = False
        for n in range(len(alt) - 1):
            if alt[n + 1][0] - alt[n][0] < MIN_PHASE:
                del alt[n: n + 2]
                changed = True
                break
        if changed:
            continue
        for n in range(len(alt) - 2):
            if alt[n + 2][0] - alt[n][0] < MIN_CYCLE:
                del alt[n + 1: n + 3]
                changed = True
                break
    return pd.DataFrame([(idx[i], k) for i, k in alt], columns=["date", "kind"])
