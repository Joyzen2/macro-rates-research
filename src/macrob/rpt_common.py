"""第 19–22 步（四篇研究报告）共用的函数：读数、合约到期日、主力合约、Newey-West t、两种安慰剂、写结果和画图。

数据在 data/reports/（公开数据的快照，不进公开仓库）；结果写到 results/<报告名>_*.csv，图写到 docs/figures/svg/。
"""
from __future__ import annotations

import calendar
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS, CHARTS = ROOT / "data" / "reports", ROOT / "results", ROOT / "docs" / "figures" / "svg"
for figures in (ROOT / "docs" / "figures", ROOT.parents[1] / "作品集" / "figures"):      # 画图库 charts.py
    if figures.exists():
        sys.path.insert(0, str(figures))

INDEX_OF = {"IF": "沪深300", "IH": "上证50", "IC": "中证500", "IM": "中证1000"}
EQUITY = ["IF", "IH", "IC", "IM"]
BOND = ["TS", "TF", "T", "TL"]


# ---------- 读数 ----------

def load(name: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def load_futures() -> pd.DataFrame:
    f = pd.read_csv(DATA / "futures_contracts.csv", parse_dates=["date"])
    f["hold"] = pd.to_numeric(f["hold"], errors="coerce").fillna(0.0)   # 持仓量 0 或空都当 0
    for c in ["open", "high", "low", "close"]:
        f.loc[f[c] <= 0, c] = np.nan                                      # 价格为 0 是数据错误（TL2306 在 2023-04-19 一行）
    return f.sort_values(["product", "date", "contract"]).reset_index(drop=True)


# ---------- 合约到期日 ----------

def nth_friday(year: int, month: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    offset = (4 - first.weekday()) % 7                  # 周五 weekday=4
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def contract_month(contract: str) -> tuple[int, int]:
    code = "".join(c for c in contract if c.isdigit())[-4:]
    return 2000 + int(code[:2]), int(code[2:])


def rule_expiry(contract: str, product: str, calendar_days: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """按规则：股指第三个周五，国债第二个周五；如果这天在已知交易日历的范围里又不是交易日，顺延到下一个交易日。"""
    y, m = contract_month(contract)
    d = pd.Timestamp(nth_friday(y, m, 2 if product in BOND else 3))
    if calendar_days is not None and len(calendar_days) and d <= calendar_days[-1] and d not in calendar_days:
        later = calendar_days[calendar_days > d]
        if len(later):
            d = later[0]
    return d


def expiry_table(fut: pd.DataFrame) -> pd.DataFrame:
    """每个合约的到期日：已到期的用数据里的最后一个交易日，还在交易的按规则算。"""
    last_day = fut["date"].max()
    rows = []
    for (prod, c), g in fut.groupby(["product", "contract"]):
        cal = pd.DatetimeIndex(sorted(fut.loc[fut["product"] == prod, "date"].unique()))
        last = g["date"].max()
        rule = rule_expiry(c, prod, cal)
        expired = last < last_day
        rows.append({"product": prod, "contract": c, "last_date": last, "rule_expiry": rule,
                     "expired": expired, "expiry": last if expired else rule})
    return pd.DataFrame(rows)


def complete_from(fut: pd.DataFrame, product: str) -> pd.Timestamp:
    """合约面板完整的起点：此后每天合约数等于常见值、且至少两个合约持仓量为正。早期数据有缺口时，从最后一个缺口的下一天起用。"""
    g = fut[fut["product"] == product]
    c = g.groupby("date").agg(n=("contract", "nunique"), pos=("hold", lambda h: int((h > 0).sum())))
    bad = c[(c.n < c.n.mode()[0]) | (c.pos < 2)]
    return c.index[c.index > bad.index.max()][0] if len(bad) else c.index[0]


def dominant(fut: pd.DataFrame, product: str, min_days: int = 0) -> pd.DataFrame:
    """每天持仓量最大的合约；min_days>0 时只在剩余自然日不少于 min_days 的合约里选。持仓相同取到期早的。"""
    exp = expiry_table(fut[fut["product"] == product]).set_index("contract")["expiry"]
    g = fut[fut["product"] == product].copy()
    g["expiry"] = g["contract"].map(exp)
    g["days"] = (g["expiry"] - g["date"]).dt.days
    if min_days:
        g = g[g["days"] >= min_days]
    g = g.sort_values(["date", "hold", "expiry"], ascending=[True, False, True])
    return g.groupby("date").head(1).set_index("date")


def price_panel(fut: pd.DataFrame, product: str) -> pd.DataFrame:
    """日期 × 合约的收盘价。"""
    g = fut[fut["product"] == product]
    return g.pivot(index="date", columns="contract", values="close").sort_index()


# ---------- 统计 ----------

def nw_ols(y, x, lags: int) -> dict:
    """y = a + b x，Newey-West（Bartlett 核）t。"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    X = sm.add_constant(x[ok])
    r = sm.OLS(y[ok], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"a": r.params[0], "b": r.params[1], "t_b": r.tvalues[1], "t_a": r.tvalues[0], "r2": r.rsquared, "n": int(ok.sum())}


def nw_mean_t(y, lags: int) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    r = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues[0])


def mean_t(y) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 2 or np.std(y, ddof=1) == 0:
        return float("nan")
    return float(np.mean(y) / (np.std(y, ddof=1) / np.sqrt(len(y))))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def shift_placebo(signal, target, stat=spearman, min_shift: int = 6) -> tuple[float, float, int]:
    """平移安慰剂：信号循环平移 min_shift..n-min_shift 期，全部算一遍；p = |统计量| 不小于真实值的比例。"""
    s, t = np.asarray(signal, float), np.asarray(target, float)
    n = len(s)
    real = stat(s, t)
    shifts = range(min_shift, n - min_shift + 1)
    vals = np.array([stat(np.roll(s, k), t) for k in shifts])
    if len(vals) == 0:
        return real, float("nan"), 0
    return real, float(np.mean(np.abs(vals) >= abs(real) - 1e-12)), len(vals)


def random_window_placebo(values_by_width: dict[int, np.ndarray], widths: list[int], real_mean: float,
                          reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """随机窗口安慰剂。values_by_width[w] 是所有可用（已避开事件）的 w 期窗口的变化；widths 是每次事件的窗口宽度。
    每次抽和事件同样多、同样宽的窗口取平均，重复 reps 次；p = |安慰剂均值 − 安慰剂均值的均值| 不小于 |真实 − 同一中心| 的比例。"""
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        draws = [values_by_width[w][rng.integers(len(values_by_width[w]))] for w in widths]
        means[i] = np.mean(draws)
    center = means.mean()
    p = float(np.mean(np.abs(means - center) >= abs(real_mean - center) - 1e-12))
    return p, float(center)


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def ann_return(nav: pd.Series) -> float:
    days = (nav.index[-1] - nav.index[0]).days
    return float((nav.iloc[-1] / nav.iloc[0]) ** (365 / days) - 1)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


# ---------- 输出 ----------

def save_csv(df: pd.DataFrame, report: str, name: str, index: bool = False) -> Path:
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"{report}_{name}.csv"
    df.to_csv(p, index=index, float_format="%.6g")
    return p


def save_svg(svg: str, name: str) -> Path:
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / f"{name}.svg"
    p.write_text(svg, encoding="utf-8")
    return p


def last_day_of_month(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.year, ts.month, calendar.monthrange(ts.year, ts.month)[1])
