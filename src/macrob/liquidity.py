"""第 17 步：资金面跟踪。运行：uv run python -m macrob.liquidity [周五日期，默认最近的周五] [--no-update]

规则（看结果之前定的，写在计划文件第 17 步里）：
- 资金价格：DR001/DR007/DR014（中国货币网图表数据，只给最近约三个月，每次运行增量存下来）；更早的用银银间回购
  定盘利率 FDR001/FDR007 代替；回购定盘利率 FR001/FR007；SHIBOR 隔夜、1 周、3 个月；交易所 GC001 收盘。
- 政策利率：7 天逆回购利率。隔夜逆回购 2026-06-29 启用，利率没有公布。
- 央行操作：华尔街见闻 7×24 快讯（robots.txt 允许），解析每天 7 天、14 天、隔夜逆回购的操作量和当天到期量；
  搜索只能回溯约半年，每次运行增量存下来。人民银行官网的 robots.txt 不允许程序访问，不抓。
- 每日净投放 = 当天各期限逆回购操作量之和 − 当天到期量；按周加总，和快讯周五报的「本周净投放」核对。
- 位置：本周均值和上周比，和过去 250 个交易日的分位比（DR 之前的日子用 FDR 拼接）。
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from .datalayer import RESULTS
from .probe_util import RAW

UA = {"User-Agent": "Mozilla/5.0"}
DR_URL = "https://www.chinamoney.com.cn/r/cms/www/chinamoney/data/currency/prr-chrt.csv"
WSCN_SEARCH = "https://api-one-wscn.awtmt.com/apiv1/search/live?query={q}&limit=20&cursor={c}"
FLASHES = RAW / "wscn_reverse_repo_flashes.jsonl"

# 7 天逆回购利率，生效日（和 data/policy_rate_changes.csv 一致；两次发布会宣布的降息分别在 09-27、05-08 执行）
RATE7 = [("2023-08-15", 1.80), ("2024-07-22", 1.70), ("2024-09-27", 1.50), ("2025-05-08", 1.40)]
OVERNIGHT_START = pd.Timestamp("2026-06-29")


# ── 取数（每次增量更新，缓存在 data/raw/）──────────────────────────────

def _get(url: str, tries: int = 3) -> bytes:
    err = None
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read()
        except Exception as e:
            err = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url[:60]}: {type(err).__name__} {str(err)[:100]}")


def _retry(fetch, tries: int = 5):
    """公开接口（经本机代理）偶尔断开：重试几次，间隔逐次拉长。"""
    for k in range(tries):
        try:
            return fetch()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def parse_dr_csv(text: str) -> pd.DataFrame:
    """中国货币网的图表数据：每行是日期，几个空字段，然后 DR001、DR007、DR014。"""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0]):
            continue
        vals = [float(p) for p in parts[1:] if p]
        if len(vals) == 3:
            rows.append((pd.Timestamp(parts[0]), *vals))
    return pd.DataFrame(rows, columns=["date", "DR001", "DR007", "DR014"]).set_index("date").sort_index()


def _merge_cache(path, new: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(path, parse_dates=["date"]).set_index("date") if path.exists() else new.iloc[:0]
    out = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
    out.to_csv(path, index_label="date")
    return out


def dr_daily(update: bool = True) -> pd.DataFrame:
    path = RAW / "chinamoney_dr.csv"
    if update:
        return _merge_cache(path, parse_dr_csv(_get(DR_URL).decode("utf-8-sig")))
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def fixings_daily(update: bool = True) -> pd.DataFrame:
    """FR001/FR007/FR014、FDR001/FDR007/FDR014；缓存和 sources.repo7d_daily 是同一个文件，这里只补最近的。"""
    path = RAW / "repo_rate_hist.csv"
    old = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if update:
        import akshare as ak
        start = (old.index.max() - pd.Timedelta(days=14)).strftime("%Y%m%d")
        new = _retry(lambda: ak.repo_rate_hist(start_date=start, end_date=pd.Timestamp.today().strftime("%Y%m%d")))
        new = new.assign(date=pd.to_datetime(new["date"])).set_index("date")
        old = _merge_cache(path, new[old.columns])
    return old


def shibor_daily(update: bool = True) -> pd.DataFrame:
    path = RAW / "shibor_daily.csv"
    if update:
        import akshare as ak
        cols = {}
        for name, ind in (("ON", "隔夜"), ("1W", "1周"), ("3M", "3月")):
            d = _retry(lambda: ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator=ind))
            cols[name] = pd.Series(pd.to_numeric(d["利率"], errors="coerce").to_numpy(), index=pd.to_datetime(d["报告日"]))
        return _merge_cache(path, pd.DataFrame(cols))
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def gc001_daily(update: bool = True) -> pd.Series:
    path = RAW / "gc001_daily.csv"
    if update:
        import akshare as ak
        d = _retry(lambda: ak.bond_buy_back_hist_em(symbol="204001"))
        d = pd.DataFrame({"GC001": pd.to_numeric(d["收盘"], errors="coerce").to_numpy()}, index=pd.to_datetime(d["日期"]))
        return _merge_cache(path, d)["GC001"]
    if not path.exists():
        return pd.Series(dtype=float, name="GC001")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")["GC001"]


def _tolerant(name: str, fetch, update: bool):
    """更新失败就用本地缓存，并把没更新上的来源记下来，写进周报的说明。"""
    if update:
        try:
            return fetch(True), None
        except Exception as e:
            print(f"{name} 没有更新上（{type(e).__name__}），用本地缓存")
            return fetch(False), name
    return fetch(False), None


def treasury_daily(update: bool = True) -> pd.DataFrame:
    """中债国债收益率曲线 1 年、10 年；补最近的到 chinabond_curve_<年>.csv。"""
    year = pd.Timestamp.today().year
    path = RAW / f"chinabond_curve_{year}.csv"
    if update:
        import akshare as ak
        old = pd.read_csv(path) if path.exists() else None
        new = _retry(lambda: ak.bond_china_yield(start_date=f"{year}0101", end_date=pd.Timestamp.today().strftime("%Y%m%d")))
        new["日期"] = pd.to_datetime(new["日期"]).dt.strftime("%Y-%m-%d")
        both = pd.concat([old, new]) if old is not None else new
        both.drop_duplicates(["曲线名称", "日期"], keep="last").sort_values(["日期", "曲线名称"]).to_csv(path, index=False)
    c = pd.concat([pd.read_csv(RAW / f"chinabond_curve_{y}.csv") for y in range(2024, year + 1)])
    c = c[c["曲线名称"] == "中债国债收益率曲线"].assign(日期=lambda d: pd.to_datetime(d["日期"]))
    return c.drop_duplicates("日期").set_index("日期").sort_index()[["1年", "10年"]].astype(float).rename(columns={"1年": "y1", "10年": "y10"})


def fetch_flashes(update: bool = True, max_pages: int = 40) -> list[dict]:
    """华尔街见闻快讯搜索「逆回购」，按 id 增量存；一页 20 条，翻到没有结果或整页都存过为止。"""
    have = {}
    if FLASHES.exists():
        for line in FLASHES.read_text(encoding="utf-8").splitlines():
            it = json.loads(line)
            have[it["id"]] = it
    if update:
        for c in range(1, max_pages + 1):
            d = json.loads(_get(WSCN_SEARCH.format(q=urllib.parse.quote("逆回购"), c=c)))["data"]
            items = d.get("items") or []
            new = [it for it in items if it["id"] not in have]
            for it in new:
                have[it["id"]] = {"id": it["id"], "display_time": it["display_time"], "text": it.get("content_text") or ""}
            if not items or not new or not d.get("next_cursor"):
                break
            time.sleep(1.0)
        FLASHES.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in sorted(have.values(), key=lambda v: v["display_time"])),
                           encoding="utf-8")
    return sorted(have.values(), key=lambda v: v["display_time"])


# ── 解析央行操作 ─────────────────────────────────────────────────────

NUM = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|零"                     # 2,150 这种千分位、「零亿元」都有
AMT = r"(" + NUM + r")\s*(万亿|亿)元(?:人民币)?"
AMT_NC = r"(?:" + NUM + r")\s*(?:万亿|亿)元(?:人民币)?"
TERM = {"d7": 7, "d14": 14}                                        # 隔夜的下一个交易日到期


def _amt(num: str, unit: str) -> float:
    return 0.0 if num == "零" else float(num.replace(",", "")) * (10000 if unit == "万亿" else 1)


def parse_flash(text: str, day: pd.Timestamp | None = None) -> dict | None:
    """一条快讯里的逆回购操作（亿元）：7 天、14 天、隔夜的操作量，当天到期量，当天净投放，7 天利率，周五报的本周净投放。
    day 是快讯的日期：快讯里写了别的日子（第二天早上转述前一天的报道），就不算这一天的。"""
    t = re.sub(r"\s+", "", text)
    if not re.match(r"(?:【[^】]*】)?(?:中国央行|央行|中国人民银行|人民银行)", t) or re.search(r"美联储|美元", t):
        return None                                        # 只认以央行开头的操作快讯，评论里转述的数不算
    if re.search(r"（(?:上证报|中证报|证券时报|证券日报|澎湃|财联社|21世纪)|流动性投放情况|货币政策执行报告", t):
        return None                                        # 报纸的综述、月度投放情况、货币政策报告
    if day is not None and any((int(m), int(dd)) != (day.month, day.day) for m, dd in re.findall(r"(\d{1,2})月(\d{1,2})日", t)):
        return None
    t = "。".join(s for s in re.split(r"[。；]", t) if not re.search(r"将在|将于|不超过", s))   # 预告的句子去掉
    if "逆回购" not in t:
        return None
    out: dict = {}
    for key, tenor in (("d7", r"(?:7|七)天期?"), ("d14", r"(?:14|十四)天期?"), ("on", r"隔夜")):
        m = re.search(AMT + tenor + r"逆回购操作", t)
        if m:
            out[key] = _amt(m.group(1), m.group(2))
    if re.search(r"(?:7|七)天期?逆回购操作量为零", t):
        out["d7"] = 0.0
    if "d7" in out:
        m = re.search(r"(?:7|七)天期?逆回购操作[，,]?(?:操作)?利率(?:为)?(\d\.\d+)%", t)
        if m:
            out["rate7"] = float(m.group(1))
    due = re.findall(r"(?:今日|当日|另)(?:有)?((?:" + AMT_NC + r"(?:7天期|七天期|14天期|隔夜)?(?:逆回购)?[和、及]?)+)到期", t)
    if due:                                                # 一条里分开说 7 天和隔夜到期的，加起来
        out["maturity"] = sum(_amt(a, u) for g in due for a, u in re.findall(AMT, g))
    elif re.search(r"今日无逆回购到期", t):
        out["maturity"] = 0.0
    m = re.search(r"本周净(投放|回笼)" + AMT, t)
    if m:
        out["week_net"] = _amt(m.group(2), m.group(3)) * (1 if m.group(1) == "投放" else -1)
    m = re.search(r"净(投放|回笼)" + AMT, re.sub(r"本周净(?:投放|回笼)" + AMT_NC, "", t))
    if m:
        out["day_net"] = _amt(m.group(2), m.group(3)) * (1 if m.group(1) == "投放" else -1)
    return out or None


def _model_maturity(d: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """按之前的操作推每天的到期量：7 天、14 天的到期日是操作日加 7、14 天（遇非交易日顺延），隔夜是下一个交易日。
    样本开头 15 天、以及没解析出操作的交易日会到期的那几天，推算不完整，记为缺。"""
    cal = calendar.sort_values().unique()
    roll = lambda x: cal[cal.searchsorted(x)] if cal.searchsorted(x) < len(cal) else pd.NaT
    due = pd.Series(0.0, index=d.index.union(pd.DatetimeIndex(cal[cal >= d.index.min()])))
    for day, r in d.iterrows():
        for key, n in TERM.items():
            if r[key] > 0:
                x = roll(day + pd.Timedelta(days=n))
                if x is not pd.NaT:
                    due[x] = due.get(x, 0.0) + r[key]
        if r["on"] > 0:
            x = roll(day + pd.Timedelta(days=1))
            if x is not pd.NaT:
                due[x] = due.get(x, 0.0) + r["on"]
    start = d.index.min()
    affected = set()
    for t in cal[(cal >= start) & (cal <= d.index.max())]:
        if t not in d.index:                                # 这一天的操作不知道：它会到期的那几天都推不全
            affected |= {roll(t + pd.Timedelta(days=1)), roll(t + pd.Timedelta(days=7)), roll(t + pd.Timedelta(days=14))}
    complete = [x - pd.Timedelta(days=15) >= start and x not in affected for x in due.index]
    return due.where(complete)


def omo_daily(flashes: list[dict], calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """按北京时间的日期合并：每个字段取当天最早报出的那条，后面的快讯只补缺。
    到期量优先用按之前操作推算的（快讯偶尔报错，比如 2026-07-29 报成前一天的 2530 亿），推不出来再用快讯报的，
    再不行用当天净投放倒推。"""
    rows: dict = {}
    for it in sorted(flashes, key=lambda v: v["display_time"]):
        day = pd.Timestamp(it["display_time"], unit="s", tz="UTC").tz_convert("Asia/Shanghai").normalize().tz_localize(None)
        p = parse_flash(it["text"], day)
        if not p:
            continue
        r = rows.setdefault(day, {"flashes": 0})
        r["flashes"] += 1
        for k, v in p.items():
            r.setdefault(k, v)
    d = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    for k in ("d7", "d14", "on", "maturity", "day_net", "rate7", "week_net"):
        if k not in d:
            d[k] = float("nan")
    d = d.rename(columns={"maturity": "maturity_reported"})
    d[["d7", "d14", "on"]] = d[["d7", "d14", "on"]].fillna(0.0)    # 快讯只报当天做了的期限，没提的就是没做
    d["ops"] = d[["d7", "d14", "on"]].sum(axis=1)
    model = _model_maturity(d, calendar if calendar is not None else d.index).reindex(d.index)
    d["maturity_model"] = model
    d["maturity"] = model.fillna(d["maturity_reported"]).fillna(d["ops"] - d["day_net"])
    d["maturity_src"] = (model.notna().map({True: "推算"}).fillna(d["maturity_reported"].notna().map({True: "快讯"}))
                         .fillna(d["day_net"].notna().map({True: "净投放倒推", False: "缺"})))
    d["net"] = d["ops"] - d["maturity"]
    d.index.name = "date"
    return d[["d7", "d14", "on", "ops", "maturity", "maturity_src", "maturity_reported", "maturity_model", "day_net", "net",
              "rate7", "week_net", "flashes"]]


# ── 资金价格 ─────────────────────────────────────────────────────────

def rate7_on(dates: pd.DatetimeIndex) -> pd.Series:
    steps = pd.Series([r for _, r in RATE7], index=pd.to_datetime([d for d, _ in RATE7]))
    return steps.reindex(steps.index.union(dates)).ffill().reindex(dates)


def daily_table(dr: pd.DataFrame, fx: pd.DataFrame, shibor: pd.DataFrame, gc: pd.Series, ty: pd.DataFrame) -> pd.DataFrame:
    idx = fx.index[fx.index >= "2015-01-01"].union(dr.index)
    d = pd.DataFrame(index=idx)
    d = d.join(dr).join(fx[["FR001", "FR007", "FDR001", "FDR007"]]).join(shibor.add_prefix("SHIBOR_")).join(gc).join(ty)
    d["DR001s"] = d["DR001"].fillna(d["FDR001"])           # 拼接：有 DR 用 DR，之前用定盘的 FDR
    d["DR007s"] = d["DR007"].fillna(d["FDR007"])
    d["rate7"] = rate7_on(d.index)
    d.index.name = "date"
    return d


def percentile(series: pd.Series, value: float, end: pd.Timestamp, n: int = 250) -> float:
    hist = series[series.index <= end].dropna().iloc[-n:]
    return float((hist < value).mean() * 100)


def week_summary(d: pd.DataFrame, omo: pd.DataFrame, friday: pd.Timestamp) -> dict:
    mon = friday - pd.Timedelta(days=4)
    wk, prev = d.loc[mon:friday], d.loc[mon - pd.Timedelta(days=7):friday - pd.Timedelta(days=7)]
    bp = lambda x: None if pd.isna(x) else round(float(x) * 100, 1)
    s = {"week": f"{mon.date()}~{friday.date()}", "days": [x.strftime("%Y-%m-%d") for x in wk.index]}
    for k in ("DR001", "DR007"):
        col = k + "s"
        s[k] = {"avg": round(wk[col].mean(), 4), "prev_avg": round(prev[col].mean(), 4), "chg_bp": bp(wk[col].mean() - prev[col].mean()),
                "last": round(wk[col].iloc[-1], 4), "vs_rate7_bp": bp((wk[col] - wk["rate7"]).mean()),
                "pct250": round(percentile(d[col], wk[col].mean(), friday), 1)}
    s["rate7"] = float(wk["rate7"].iloc[-1])
    s["FR001_DR001_bp"] = bp((wk["FR001"] - wk["DR001s"]).mean())
    s["GC001_DR001_bp"] = bp((wk["GC001"] - wk["DR001s"]).mean())
    s["SHIBOR_3M"] = {"last": float(wk["SHIBOR_3M"].iloc[-1]), "chg_bp": bp(wk["SHIBOR_3M"].iloc[-1] - prev["SHIBOR_3M"].iloc[-1])}
    for k in ("y1", "y10"):
        s[k] = {"last": float(wk[k].iloc[-1]), "chg_bp": bp(wk[k].iloc[-1] - prev[k].iloc[-1])}
    o = omo.loc[mon:friday]
    s["omo"] = {"days": {x.strftime("%m-%d"): {k: (None if pd.isna(v) else float(v)) for k, v in r[["d7", "d14", "on", "maturity", "net"]].items()}
                         for x, r in o.iterrows()},
                "ops": float(o["ops"].sum()), "maturity": float(o["maturity"].sum()), "net": float(o["net"].sum()),
                "reported_week_net": None if o["week_net"].dropna().empty else float(o["week_net"].dropna().iloc[-1]),
                "missing_days": [x.strftime("%m-%d") for x in wk.index if x not in o.index or pd.isna(o.loc[x, "net"])],
                "maturity_mismatch": {x.strftime("%m-%d"): [float(r["maturity_reported"]), float(r["maturity_model"])] for x, r in o.iterrows()
                                      if pd.notna(r["maturity_reported"]) and pd.notna(r["maturity_model"])
                                      and abs(r["maturity_reported"] - r["maturity_model"]) > 0.5}}
    overlap = d[["DR001", "FDR001", "DR007", "FDR007"]].dropna()
    s["dr_vs_fdr_bp"] = {"days": len(overlap), "DR001": bp((overlap["DR001"] - overlap["FDR001"]).abs().mean()),
                         "DR007": bp((overlap["DR007"] - overlap["FDR007"]).abs().mean())}
    return s


def weekly_table(d: pd.DataFrame, omo: pd.DataFrame) -> pd.DataFrame:
    w = d[["DR001s", "DR007s", "rate7", "FR001", "GC001"]].resample("W-FRI").mean()
    w["DR001_vs_rate7_bp"] = (w["DR001s"] - w["rate7"]) * 100
    w["DR007_vs_rate7_bp"] = (w["DR007s"] - w["rate7"]) * 100
    o = omo[["ops", "maturity", "net"]].resample("W-FRI").sum(min_count=1)
    o["reported_week_net"] = omo["week_net"].resample("W-FRI").last()
    return w.join(o).dropna(subset=["DR001s"])


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    update = "--no-update" not in sys.argv
    stale = []
    fetched = {}
    for name, fn in (("DR", dr_daily), ("定盘利率", fixings_daily), ("SHIBOR", shibor_daily), ("GC001", gc001_daily),
                     ("国债收益率", treasury_daily), ("快讯", fetch_flashes)):
        fetched[name], miss = _tolerant(name, fn, update)
        stale += [miss] if miss else []
    dr, fx, sh, gc, ty, flashes = (fetched[k] for k in ("DR", "定盘利率", "SHIBOR", "GC001", "国债收益率", "快讯"))
    d = daily_table(dr, fx, sh, gc, ty)
    omo = omo_daily(flashes, d.index)
    last = d.dropna(subset=["DR001s"]).index.max()
    friday = pd.Timestamp(args[0]) if args else last - pd.Timedelta(days=(last.weekday() - 4) % 7)
    d.to_csv(RESULTS / "liquidity_daily.csv")
    omo.to_csv(RESULTS / "liquidity_omo_daily.csv")
    weekly_table(d, omo).round(4).to_csv(RESULTS / "liquidity_weekly.csv")
    s = week_summary(d, omo, friday)
    s["stale_sources"] = stale
    (RESULTS / f"liquidity_week_{friday:%Y%m%d}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"快讯 {len(flashes)} 条，解析出 {len(omo)} 天的操作（{omo.index.min():%Y-%m-%d} 起）；DR {dr.index.min():%Y-%m-%d} 至 {dr.index.max():%Y-%m-%d}")
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
