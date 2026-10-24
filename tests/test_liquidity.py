"""第 17 步资金面的关键规则：快讯里央行操作的各种说法、到期量的推算、每周净投放能和快讯报的对上。"""
import pandas as pd
import pytest

from macrob import liquidity as L

DAY = pd.Timestamp("2026-10-23")


@pytest.mark.parametrize("text, expected", [
    ("中国央行公开市场今日七天期逆回购操作量为零，进行2580亿元人民币隔夜逆回购操作。今日2330亿元逆回购到期。中国央行公开市场本周净投放620亿元。",
     {"d7": 0, "on": 2580, "maturity": 2330, "week_net": 620}),
    ("中国央行进行2,150亿元7天期逆回购操作，操作利率1.4%，持平上次。今日1230亿元逆回购到期。中国央行公开市场本周净回笼6827亿元。",
     {"d7": 2150, "rate7": 1.4, "maturity": 1230, "week_net": -6827}),
    ("中国央行今日开展695亿元7天期逆回购操作，操作利率1.4%，与此前持平。今日有5245亿元7天期逆回购到期。"
     "同时，中国央行进行6000亿元隔夜逆回购操作，今日有3000亿元隔夜逆回购到期。",
     {"d7": 695, "on": 6000, "maturity": 8245}),                       # 分两句说的到期量要加起来
    ("中国央行今日开展1000亿元7天期逆回购操作，操作利率1.4%，与此前持平。今日6625亿元7天期和6000亿元隔夜逆回购到期。",
     {"d7": 1000, "maturity": 12625}),
    ("中国央行公开市场开展零亿元7天期逆回购操作。今日1776亿元逆回购到期。", {"d7": 0, "maturity": 1776}),
    ("中国央行连续7个工作日7天期逆回购操作量为零，今日开展3274亿元人民币隔夜逆回购操作，另有4697亿元人民币隔夜逆回购到期。",
     {"d7": 0, "on": 3274, "maturity": 4697}),
    ("中国央行今日开展0亿元7天期逆回购操作，今日净回笼5985亿元。", {"d7": 0, "day_net": -5985}),
    ("中国央行今日开展3255亿元7天逆回购操作，操作利率为1.40%。今日有3985亿元逆回购到期，4000亿元MLF到期。",
     {"d7": 3255, "rate7": 1.4, "maturity": 3985}),                    # MLF 到期不算进逆回购
])
def test_parse_flash_handles_the_wordings_seen(text, expected):
    p = L.parse_flash(text, DAY)
    assert {k: p[k] for k in expected} == pytest.approx(expected)


@pytest.mark.parametrize("text, day", [
    ("周五（10月23日），美联储隔夜逆回购协议（RRP）使用规模为2亿美元（交易对手1家），上个交易日报3.51亿美元。", DAY),
    ("中国人民银行：将在10月21日至11月2日开展隔夜逆回购操作，采用固定利率、数量招标，每日操作量不超过10000亿元。", DAY),
    ("央行逆回购再现“零操作”。8月11日，央行未开展7天期逆回购操作，当日有465亿元逆回购到期，公开市场实现净回笼465亿元。",
     pd.Timestamp("2026-08-12")),                                     # 第二天早上转述前一天的报道
    ("资金价格持续下探……中国央行今日开展500亿元7天期逆回购操作。", DAY),     # 评论里转述的数不算
])
def test_parse_flash_skips_fed_previews_and_recaps(text, day):
    assert L.parse_flash(text, day) is None


def _flash(day: str, text: str, hour: int = 9) -> dict:
    ts = pd.Timestamp(f"{day} {hour:02d}:20", tz="Asia/Shanghai").tz_convert("UTC")
    return {"id": hash((day, text)), "display_time": int(ts.timestamp()), "text": text}


def test_model_maturity_rolls_over_holidays_and_overnight():
    cal = pd.DatetimeIndex(["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18",
                            "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25",
                            "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"])
    d = pd.DataFrame({"d7": 0.0, "d14": 0.0, "on": 0.0}, index=cal[:13])
    d.loc["2026-09-14", "d7"] = 100          # 7 天后 09-21 到期
    d.loc["2026-09-18", "on"] = 200          # 周五的隔夜，下周一 09-21 到期
    d.loc["2026-09-29", "d7"] = 300          # 7 天后是 10-06，假期，顺延到 10-08
    d.loc["2026-09-30", "on"] = 400          # 节前最后一天的隔夜，节后第一天 10-08 到期
    due = L._model_maturity(d, cal)
    assert pd.isna(due["2026-09-21"])                                  # 样本开头 15 天，推不全
    d2 = pd.concat([pd.DataFrame({"d7": 0.0, "d14": 0.0, "on": 0.0}, index=pd.date_range("2026-08-24", "2026-09-11", freq="B")), d])
    full = L._model_maturity(d2, cal.union(d2.index))
    assert full["2026-09-21"] == 300
    assert full["2026-10-08"] == 700


def test_week_net_reconciles_with_reported():
    texts = {
        "2026-10-19": "中国央行公开市场开展800亿元7天期逆回购操作，操作利率1.40%，持平上次。今日500亿元逆回购到期。",
        "2026-10-20": "中国央行公开市场今日七天期逆回购操作量为零。今日650亿元逆回购到期。",
        "2026-10-21": "中国央行公开市场今日七天期逆回购操作量为零，进行800亿元人民币隔夜逆回购操作。今日560亿元逆回购到期。",
        "2026-10-22": "中国央行公开市场今日七天期逆回购操作量为零，进行1730亿元人民币隔夜逆回购操作。今日1250亿元逆回购到期。",
        "2026-10-23": "中国央行公开市场今日七天期逆回购操作量为零，进行2580亿元人民币隔夜逆回购操作。今日2330亿元逆回购到期。"
                      "中国央行公开市场本周净投放620亿元。",
    }
    omo = L.omo_daily([_flash(d, t) for d, t in texts.items()], pd.DatetimeIndex(list(texts)))
    assert omo["net"].tolist() == [300, -650, 240, 480, 250]
    assert omo["net"].sum() == omo["week_net"].dropna().iloc[-1] == 620


def test_dr_csv_and_policy_rate_steps():
    dr = L.parse_dr_csv("2026-10-23,,,,,,1.2737,1.3214,1.343\n2026-10-22,,,,,,1.2619,1.3179,1.3654\n")
    assert dr.loc["2026-10-23", "DR001"] == 1.2737 and dr.index.is_monotonic_increasing
    r = L.rate7_on(pd.DatetimeIndex(["2024-07-19", "2024-07-22", "2025-05-07", "2025-05-08"]))
    assert r.tolist() == [1.80, 1.70, 1.50, 1.40]
    csv = pd.read_csv(L.RAW.parents[0] / "policy_rate_changes.csv")
    assert csv.loc[csv["date"] == "2024-07-22", "new_rate"].iloc[0] == 1.70
