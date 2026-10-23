"""两状态马尔可夫区制转换模型（扩张 / 收缩），均值和方差都随状态切换。

- 平滑概率：用全样本估计参数，并用前后所有数据推断每个月的状态。带前视，只作对照。
- 实时滤波概率：每年年末用到当时为止的数据重估参数，之后 12 个月用这组参数做滤波，
  每个月只用到当月为止的数据。这是实际能用的版本。
- 状态命名：均值高的那个状态叫扩张。每次重估都重新判断，避免两个状态的编号对调。
- 状态切换：扩张概率穿过 0.5，而且新状态至少持续 3 个月，才算一次切换；日期默认记在新状态的第一个月，
  修正后的版本记在确认的那个月（见 switches）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

MIN_FIT, PERSIST = 60, 3


def _model(y: pd.Series) -> MarkovRegression:
    return MarkovRegression(y.to_numpy(dtype=float), k_regimes=2, trend="c", switching_variance=True)


FAILURES = (np.linalg.LinAlgError, ValueError, RuntimeError)


def _fit(y: pd.Series):
    """随机初值偶尔会把一个状态压空，导致矩阵分解失败，或者转移矩阵退化、算不出稳态概率：
    换初值重试，并确认这组参数能拿来滤波；都失败就抛出，由调用方沿用上一次的参数。"""
    last = None
    for reps in (20, 0, 50):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = _model(y).fit(disp=False, search_reps=reps, rng=reps)    # statsmodels 0.15 不看 np.random.seed，种子要直接传
                _model(y).filter(res.params)
                return res
            except FAILURES as e:
                last = e
    raise last


def _expansion(res) -> int:
    names = list(res.model.param_names)
    return int(np.argmax([res.params[names.index(f"const[{k}]")] for k in range(2)]))


def smoothed(y: pd.Series) -> pd.Series:
    res = _fit(y)
    return pd.Series(res.smoothed_marginal_probabilities[:, _expansion(res)], index=y.index, name="平滑概率")


def realtime(y: pd.Series) -> pd.Series:
    out = {}
    fit_ends = [t for t in y.index if t.month == 12 and y.index.get_loc(t) + 1 >= MIN_FIT]
    res = None
    for j, T in enumerate(fit_ends):
        try:
            res = _fit(y.loc[:T])
        except FAILURES:
            if res is None:
                continue                                    # 第一次就失败：这一年不出数
        k = _expansion(res)
        nxt = fit_ends[j + 1] if j + 1 < len(fit_ends) else y.index[-1]
        for m in y.loc[T:nxt].index[1:] if j + 1 < len(fit_ends) else y.loc[T:].index[1:]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    f = _model(y.loc[:m]).filter(res.params)
                except FAILURES:
                    continue                                # 这个月滤波失败：不出数（沿用前一个月的由下游决定）
            out[m] = float(f.filtered_marginal_probabilities[-1, k])
    return pd.Series(out, name="实时滤波概率").sort_index()


def switches(p: pd.Series, confirm: bool = False) -> pd.DataFrame:
    """概率序列 → 状态切换：date, kind（peak = 扩张转收缩，trough = 收缩转扩张）。

    新状态要连续 PERSIST 个月才算切换。confirm=False 把日期记在第 1 个月（原来的做法）；
    修正（看完结果后发现）：实时只能在第 PERSIST 个月确认，confirm=True 把日期记在确认的那个月。
    """
    state = (p >= 0.5).astype(int).to_numpy()
    idx, rows, cur = p.index, [], state[0]
    i = 1
    while i < len(state):
        if state[i] != cur and (state[i: i + PERSIST] == state[i]).all() and len(state[i: i + PERSIST]) == PERSIST:
            rows.append((idx[i + PERSIST - 1] if confirm else idx[i], "peak" if cur == 1 else "trough"))
            cur = state[i]
        i += 1
    return pd.DataFrame(rows, columns=["date", "kind"])
