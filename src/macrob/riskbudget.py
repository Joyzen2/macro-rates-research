"""协方差估计、风险预算求解、带约束再平衡。

1) Ledoit-Wolf：股、债、商品的波动率差一个数量级（周波动约 3% / 0.3% / 1.5%），
   直接对协方差矩阵向「平均方差 × 单位阵」收缩，会把债券方差拉高几十倍。
   所以先把各资产收益标准化，对相关系数矩阵做 LW 收缩（相关系数向 0 收缩），
   再乘回各自的样本波动率。收缩强度由 LW 解析公式给出，不是手调参数。

2) 风险预算：给定预算 b（各资产应承担的风险比例），求 w 使风险贡献
   RC_k = w_k (Σw)_k / (w'Σw) = b_k。等价的凸问题（Spinu 2013）：
       min_y  ½ y'Σy - Σ b_k ln y_k,   y > 0；   w = y / Σy
   一阶条件 (Σy)_k = b_k / y_k，即 y_k (Σy)_k = b_k，归一化后风险贡献恰为 b。

3) 带约束再平衡：风险预算权重是「理想目标」，执行时加换手上限和权重上下限：
       min_w (w - w*)'Σ(w - w*)   s.t.  Σw = 1,  lb ≤ w ≤ ub,  ||w - w_prev||₁ ≤ τ
   目标函数是相对理想权重的跟踪方差，用同一个协方差矩阵度量偏离。
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
from sklearn.covariance import LedoitWolf


def sample_cov(R: np.ndarray) -> np.ndarray:
    return np.cov(R, rowvar=False, ddof=1)


def ledoit_wolf_corr(R: np.ndarray) -> tuple[np.ndarray, float]:
    """对标准化收益做 LW 收缩，返回 (协方差, 收缩强度)。"""
    sd = R.std(axis=0, ddof=0)
    Z = (R - R.mean(axis=0)) / sd
    lw = LedoitWolf(assume_centered=True).fit(Z)
    C = lw.covariance_
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)                     # 化成严格的相关系数矩阵（对角线 = 1）
    sd_unbiased = R.std(axis=0, ddof=1)
    return C * np.outer(sd_unbiased, sd_unbiased), float(lw.shrinkage_)


def risk_contributions(w: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = S @ w
    return w * m / float(w @ m)


def risk_budget_weights(S: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    if np.any(b <= 0):
        raise ValueError("风险预算必须全部为正（log 障碍要求 y_k > 0）")
    b = b / b.sum()
    n = len(b)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(n))
    y = _solve(L, b)
    if y is None:      # 方差很小时 CLARABEL 数值上解不动：只在这时把协方差缩放到平均方差为 1 再解。权重和尺度无关，规则不变
        S1 = S / np.mean(np.diag(S))
        y = _solve(np.linalg.cholesky(S1 + 1e-12 * np.eye(n)), b)
    if y is None:
        raise RuntimeError("风险预算求解失败")
    w = np.maximum(y, 0)
    return w / w.sum()


def _solve(L: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    y = cp.Variable(len(b), pos=True)
    try:
        cp.Problem(cp.Minimize(0.5 * cp.sum_squares(L.T @ y) - b @ cp.log(y))).solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    return y.value


RELAXED = {"count": 0}      # 换手上限与权重上下限冲突、只能放开换手上限的次数


def constrained_rebalance(target: np.ndarray, prev: np.ndarray | None, S: np.ndarray,
                          max_turnover: float | None, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    """若上期持仓本身违反新的上下限，满足上下限所需的换手可能超过上限，问题无解；
    这时优先保证上下限、放开换手上限，并计数。"""
    n = len(target)
    L = np.linalg.cholesky(S + 1e-12 * np.eye(n))

    def solve(with_turnover: bool):
        w = cp.Variable(n)
        cons = [cp.sum(w) == 1, w >= lb, w <= ub]
        if with_turnover:
            cons.append(cp.norm1(w - prev) <= max_turnover)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ (w - target))), cons)
        prob.solve(solver=cp.CLARABEL)
        return w.value if prob.status in ("optimal", "optimal_inaccurate") else None

    use_t = prev is not None and max_turnover is not None
    x = solve(use_t)
    if x is None and use_t:
        RELAXED["count"] += 1
        x = solve(False)
    if x is None:
        raise RuntimeError("再平衡求解失败")
    x = np.clip(x, lb, ub)
    return x / x.sum()
