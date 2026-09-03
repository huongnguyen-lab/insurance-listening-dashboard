import json
import math
import os
import unicodedata
import pandas as pd
from datetime import timedelta
from functools import lru_cache


def _topic_slug(s: str) -> str:
    """Normalize topic for deduplication: underscoreâ†’space, strip diacritics, lowercase."""
    s = s.replace("_", " ").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def _best_topic_name(names: list) -> str:
    """Among equivalent topics, prefer the one with Vietnamese diacritics."""
    def score(n):
        nfd = unicodedata.normalize("NFD", n)
        has_marks = any(unicodedata.category(c) == "Mn" for c in nfd)
        return (has_marks, len(n))
    return max(names, key=score)


TAXONOMY_INTENT_LABELS = {
    "khen": "Khen",
    "muon_tu_van": "Muon tu van",
    "y_kien_trung_lap": "Y kien trung lap",
    "phan_van": "Phan van",
    "so_sanh_chung": "So sanh chung",
    "check_chi_phi": "Check chi phi",
    "bao_hiem_chung": "Bao hiem chung",
    "chinh_sach_quy_trinh_boi_thuong": "Chinh sach & quy trinh boi thuong",
    "san_pham_giai_phap": "San pham / giai phap",
    "dich_vu_tu_van": "Dich vu tu van",
    "spam": "Spam",
}

TAXONOMY_ORDER = {key: idx for idx, key in enumerate(TAXONOMY_INTENT_LABELS)}

NEGATIVE_TOPIC_LABELS = {
    "bao_hiem_chung": "Bao hiem chung",
    "chinh_sach_quy_trinh_boi_thuong": "Chinh sach & quy trinh boi thuong",
    "san_pham_giai_phap": "San pham / giai phap",
    "dich_vu_tu_van": "Dich vu tu van",
}

ADVISOR_DETAIL_LABELS = {
    "tu_van_sai": "Tu van sai",
    "khong_minh_bach": "Khong minh bach",
    "tu_choi_ho_tro": "Tu choi ho tro",
}


def _text_slug(value) -> str:
    text = str(value or "").replace("_", " ").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


def _taxonomy_negative_topic(*values) -> str:
    text = _text_slug(" ".join(str(v or "") for v in values))
    if any(k in text for k in ["tu van", "tvv", "dai ly", "moi gioi", "nhan vien", "ho tro", "cskh"]):
        return "dich_vu_tu_van"
    if any(k in text for k in ["boi thuong", "claim", "chi tra", "tu choi boi thuong", "ho so", "quy trinh"]):
        return "chinh_sach_quy_trinh_boi_thuong"
    if any(k in text for k in ["san pham", "pham vi", "thoi han", "thoi gian cho", "loai tru", "quyen loi", "dieu khoan", "hop dong", "goi bao hiem"]):
        return "san_pham_giai_phap"
    if any(k in text for k in ["lua dao", "kien", "kien tung", "thieu minh bach", "khong minh bach", "scam", "phot", "bao hiem chung"]):
        return "bao_hiem_chung"
    return "bao_hiem_chung"


def _advisor_details(*values) -> list[str]:
    text = _text_slug(" ".join(str(v or "") for v in values))
    details = []
    if any(k in text for k in ["tu van sai", "sai thong tin", "tu van nham", "noi sai"]):
        details.append("tu_van_sai")
    if any(k in text for k in ["khong minh bach", "thieu minh bach", "map mo", "che giau", "khong ro rang"]):
        details.append("khong_minh_bach")
    if any(k in text for k in ["tu choi ho tro", "khong ho tro", "bo mac", "lien he khong duoc", "khong phan hoi"]):
        details.append("tu_choi_ho_tro")
    return details


def _taxonomy_intent(row) -> str:
    intent = _text_slug(row.get("intent"))
    sentiment = _text_slug(row.get("sentiment"))
    text = " ".join(str(row.get(col, "") or "") for col in ["Content", "summary", "pain_points"])
    if intent == "spam" or bool(row.get("is_seeding", False)):
        return "spam"
    if intent in TAXONOMY_INTENT_LABELS:
        return intent
    if sentiment == "tieu cuc" or intent == "khieu nai":
        return _taxonomy_negative_topic(intent, text)
    if intent == "khen":
        return "khen"
    if intent == "tu van":
        return "spam" if bool(row.get("is_advisor", False)) else "muon_tu_van"
    if intent == "hoi gia":
        return "check_chi_phi"
    if intent == "so sanh":
        return "so_sanh_chung"
    if any(k in _text_slug(text) for k in ["phan van", "lan tan", "khong biet", "nen mua", "co nen"]):
        return "phan_van"
    return "y_kien_trung_lap"


def _taxonomy_topic_name(topic: str) -> str:
    slug = _text_slug(topic)
    if slug in TAXONOMY_INTENT_LABELS:
        return TAXONOMY_INTENT_LABELS[slug]
    if slug in NEGATIVE_TOPIC_LABELS:
        return NEGATIVE_TOPIC_LABELS[slug]
    if any(k in slug for k in ["tu van sai", "khong minh bach", "thieu minh bach", "map mo", "tu choi ho tro", "khong ho tro"]):
        return NEGATIVE_TOPIC_LABELS["dich_vu_tu_van"]
    if any(k in slug for k in ["boi thuong", "claim", "chi tra", "tu choi boi thuong", "ho so", "quy trinh"]):
        return NEGATIVE_TOPIC_LABELS["chinh_sach_quy_trinh_boi_thuong"]
    if any(k in slug for k in ["san pham", "pham vi", "thoi han", "thoi gian cho", "loai tru", "quyen loi", "dieu khoan", "hop dong"]):
        return NEGATIVE_TOPIC_LABELS["san_pham_giai_phap"]
    if any(k in slug for k in ["lua dao", "kien", "kien tung", "thieu minh bach", "bao hiem chung", "scam", "phot"]):
        return NEGATIVE_TOPIC_LABELS["bao_hiem_chung"]
    return str(topic).strip().lower().replace("_", " ")


def _intent_counts(df: pd.DataFrame) -> list[dict]:
    if df.empty or "intent" not in df:
        return []
    rows = df.apply(_taxonomy_intent, axis=1).value_counts().reset_index(name="count")
    rows = rows.rename(columns={"index": "intent"})
    rows["label"] = rows["intent"].map(TAXONOMY_INTENT_LABELS).fillna(rows["intent"])
    rows["order"] = rows["intent"].map(TAXONOMY_ORDER).fillna(999)
    rows = rows.sort_values(["order", "count"], ascending=[True, False])
    return rows[["intent", "label", "count"]].to_dict(orient="records")


def _to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of dicts with NaN â†’ None (JSON-safe)."""
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if not isinstance(value, (str, bytes, list, dict)) and pd.isna(value):
        return None
    return value


@lru_cache(maxsize=16)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _read_csv_cached(path: str) -> pd.DataFrame:
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    return _read_cached(path, mtime).copy()


@lru_cache(maxsize=8)
def _load_joined_cached(base_dir: str, raw_mtime: float, labels_mtime: float) -> pd.DataFrame:
    raw_path = f"{base_dir}/raw_comments.csv"
    labels_path = f"{base_dir}/ai_labels.csv"
    df_raw = _read_cached(raw_path, raw_mtime)
    df_labels = _read_cached(labels_path, labels_mtime)
    if df_raw.empty or df_labels.empty:
        return pd.DataFrame()
    df = df_raw.merge(df_labels, on="CommentID", how="inner")
    df["Date"] = _parse_comment_dates(df["Date"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
    return df


def _joined_all(base_dir: str) -> pd.DataFrame:
    raw_path = f"{base_dir}/raw_comments.csv"
    labels_path = f"{base_dir}/ai_labels.csv"
    raw_mtime = os.path.getmtime(raw_path) if os.path.exists(raw_path) else 0.0
    labels_mtime = os.path.getmtime(labels_path) if os.path.exists(labels_path) else 0.0
    return _load_joined_cached(base_dir, raw_mtime, labels_mtime).copy()


def _load(base_dir: str):
    def read(name):
        path = f"{base_dir}/{name}.csv"
        return _read_csv_cached(path)

    return read("raw_comments"), read("ai_labels"), read("topics"), read("crisis_alerts"), read("groups")


def _parse_comment_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # Crawler exports ISO dates (YYYY-MM-DD). Parse them explicitly before
    # falling back to legacy Vietnamese day-first timestamps; otherwise pandas
    # can interpret 2026-09-01 as 9 January instead of 1 September.
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            s[missing], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            s[missing], format="%H:%M:%S %d/%m/%Y", errors="coerce"
        )
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s[missing], dayfirst=True, errors="coerce")
    return parsed


def _previous_period(start_date: str, end_date: str) -> tuple[str, str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    delta = end - start
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta
    return prev_start.strftime("%Y-%m-%d"), prev_end.strftime("%Y-%m-%d")


def _periodize(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    period_map = {"day": "D", "week": "W", "month": "M", "year": "Y"}
    freq = period_map.get(granularity, "W")
    df = df.dropna(subset=["Date"]).copy()
    df["period"] = df["Date"].dt.to_period(freq).apply(lambda r: r.start_time.strftime("%Y-%m-%d"))
    return df


def _split_csv_filter(value: str | None) -> list[str]:
    if not value or str(value) == "all":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip() and v.strip() != "all"]


def _filter_insurance_lines(df: pd.DataFrame, base_dir: str, insurance_lines: str = "all") -> pd.DataFrame:
    values = _split_csv_filter(insurance_lines)
    if not values or df.empty:
        return df
    path = f"{base_dir}/insurance_lines.csv"
    if not os.path.exists(path):
        return df.iloc[0:0]
    line_df = _read_csv_cached(path)
    if line_df.empty or "CommentID" not in line_df or "insurance_line" not in line_df:
        return df.iloc[0:0]
    mask = line_df["insurance_line"].astype(str).isin(values)
    if "insurance_subline" in line_df.columns:
        mask = mask | line_df["insurance_subline"].astype(str).isin(values)
    ids = set(line_df[mask]["CommentID"].astype(str))
    return df[df["CommentID"].astype(str).isin(ids)]


def _count_unique_posts(base_dir: str, comment_ids: set[str] | None = None) -> int:
    raw_path = f"{base_dir}/raw_comments.csv"
    if not os.path.exists(raw_path):
        return 0
    raw = _read_csv_cached(raw_path)
    if raw.empty:
        return 0
    if comment_ids is not None and "CommentID" in raw:
        raw = raw[raw["CommentID"].astype(str).isin(comment_ids)]
    if raw.empty:
        return 0
    for post_col in ("PostID", "PostURL"):
        if post_col not in raw:
            continue
        post_values = (
            raw[post_col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .dropna()
        )
        if not post_values.empty:
            return int(post_values.nunique())
    if {"CommentID", "ReplyToID"}.issubset(raw.columns):
        reply_ids = set(
            raw["ReplyToID"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .dropna()
        )
        return int((~raw["CommentID"].astype(str).isin(reply_ids)).sum())
    return int(len(raw))


def _joined(df_raw, df_labels, start_date: str, end_date: str, group_id: str,
            advisor_filter: str = "all", brand_id: str = "all", base_dir: str = "./data",
            insurance_lines: str = "all"):
    """Inner-join raw_comments + ai_labels, filter by date range and group.
    advisor_filter: 'all' | 'exclude' (hide TVV) | 'only' (chá»‰ TVV)
    """
    if df_raw.empty or df_labels.empty:
        return pd.DataFrame()

    df = _joined_all(base_dir)
    if df.empty:
        return df

    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    if group_id and group_id != "all":
        df = df[df["group_id"] == group_id]
    if brand_id and brand_id != "all":
        mentions_path = f"{base_dir}/brand_mentions.csv"
        if os.path.exists(mentions_path):
            df_mentions = pd.read_csv(mentions_path)
            brand_ids = set(
                df_mentions[df_mentions["brand_id"].astype(str) == str(brand_id)]["CommentID"].astype(str)
            )
            df = df[df["CommentID"].astype(str).isin(brand_ids)]
        else:
            df = df.iloc[0:0]
    if "is_advisor" in df.columns:
        if advisor_filter == "exclude":
            df = df[df["is_advisor"] != True]
        elif advisor_filter == "only":
            df = df[df["is_advisor"] == True]
    df = _filter_insurance_lines(df, base_dir, insurance_lines)
    return df


def _slice_metrics(df, df_crisis):
    total = len(df)
    if total == 0:
        return {"total": 0, "positive_pct": 0.0, "super_negative_pct": 0.0, "crisis_count": 0}

    positive = int((df["sentiment"] == "tich_cuc").sum())
    super_neg = int(((df["sentiment"] == "tieu_cuc") & (df["sentiment_score"] <= -0.7)).sum())
    crisis_count = 0
    if not df_crisis.empty:
        crisis_count = int(df_crisis["CommentID"].isin(df["CommentID"]).sum())

    return {
        "total": total,
        "positive_pct": round(positive / total * 100, 1),
        "super_negative_pct": round(super_neg / total * 100, 1),
        "crisis_count": crisis_count,
    }


def get_metrics(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                advisor_filter: str = "all", brand_id: str = "all") -> dict:
    df_raw, df_labels, _, df_crisis, _ = _load(base_dir)

    df_curr = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)

    # Previous period: same length, immediately before start_date
    start = pd.Timestamp(start_date)
    delta = pd.Timestamp(end_date) - start
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta

    df_prev = _joined(df_raw, df_labels,
                      prev_start.strftime("%Y-%m-%d"),
                      prev_end.strftime("%Y-%m-%d"),
                      group_id, advisor_filter, brand_id, base_dir)

    curr = _slice_metrics(df_curr, df_crisis)
    prev = _slice_metrics(df_prev, df_crisis)

    return {
        "current": curr,
        "previous": prev,
        "delta": {
            "total": curr["total"] - prev["total"],
            "positive_pct": round(curr["positive_pct"] - prev["positive_pct"], 1),
            "super_negative_pct": round(curr["super_negative_pct"] - prev["super_negative_pct"], 1),
            "crisis_count": curr["crisis_count"] - prev["crisis_count"],
        },
    }


def get_sentiment_trend(start_date: str, end_date: str, group_id: str, granularity: str = "week",
                        base_dir: str = "./data", advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    df = _periodize(df.dropna(subset=["sentiment"]).copy(), granularity)

    result = []
    for period, grp in df.groupby("period"):
        result.append({
            "period": period,
            "tich_cuc": int((grp["sentiment"] == "tich_cuc").sum()),
            "tieu_cuc": int((grp["sentiment"] == "tieu_cuc").sum()),
            "trung_lap": int((grp["sentiment"] == "trung_lap").sum()),
        })
    return sorted(result, key=lambda x: x["period"])


def get_comment_timeline(start_date: str, end_date: str, group_id: str, granularity: str = "week",
                         base_dir: str = "./data", advisor_filter: str = "all",
                         brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    prev_start, prev_end = _previous_period(start_date, end_date)
    df_curr = _periodize(_joined(df_raw, df_labels, start_date, end_date, group_id,
                                 advisor_filter, brand_id, base_dir), granularity)
    df_prev = _periodize(_joined(df_raw, df_labels, prev_start, prev_end, group_id,
                                 advisor_filter, brand_id, base_dir), granularity)

    curr_counts = df_curr.groupby("period").size().reset_index(name="current").sort_values("period")
    prev_counts = df_prev.groupby("period").size().reset_index(name="previous").sort_values("period")
    max_len = max(len(curr_counts), len(prev_counts))
    rows = []
    for i in range(max_len):
        curr = curr_counts.iloc[i].to_dict() if i < len(curr_counts) else {"period": "", "current": 0}
        prev = prev_counts.iloc[i].to_dict() if i < len(prev_counts) else {"period": "", "previous": 0}
        rows.append({
            "period": curr["period"] or prev["period"],
            "previous_period": prev["period"],
            "current": int(curr["current"]),
            "previous": int(prev["previous"]),
            "delta": int(curr["current"]) - int(prev["previous"]),
        })
    return rows


def get_sentiment_comparison(start_date: str, end_date: str, group_id: str, granularity: str = "week",
                             base_dir: str = "./data", advisor_filter: str = "all",
                             brand_id: str = "all") -> list:
    prev_start, prev_end = _previous_period(start_date, end_date)
    curr = get_sentiment_trend(start_date, end_date, group_id, granularity, base_dir, advisor_filter, brand_id)
    prev = get_sentiment_trend(prev_start, prev_end, group_id, granularity, base_dir, advisor_filter, brand_id)
    max_len = max(len(curr), len(prev))
    empty = {"period": "", "tich_cuc": 0, "tieu_cuc": 0, "trung_lap": 0}
    rows = []
    for i in range(max_len):
        c = curr[i] if i < len(curr) else empty
        p = prev[i] if i < len(prev) else empty
        rows.append({
            "period": c["period"] or p["period"],
            "previous_period": p["period"],
            "tich_cuc": int(c["tich_cuc"]),
            "tieu_cuc": int(c["tieu_cuc"]),
            "trung_lap": int(c["trung_lap"]),
            "prev_tich_cuc": int(p["tich_cuc"]),
            "prev_tieu_cuc": int(p["tieu_cuc"]),
            "prev_trung_lap": int(p["trung_lap"]),
        })
    return rows


def get_sentiment_score_distribution(start_date: str, end_date: str, group_id: str,
                                     base_dir: str = "./data", advisor_filter: str = "all",
                                     brand_id: str = "all", insurance_lines: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir, insurance_lines)
    if df.empty:
        return []
    scores = pd.to_numeric(df["sentiment_score"], errors="coerce").dropna()
    bins = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ["-1.0..-0.75", "-0.75..-0.5", "-0.5..-0.25", "-0.25..0",
              "0..0.25", "0.25..0.5", "0.5..0.75", "0.75..1.0"]
    cut = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    counts = cut.value_counts().reindex(labels, fill_value=0)
    return [{"bucket": bucket, "count": int(count)} for bucket, count in counts.items()]


def get_top_topics(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                   top_n: int = 10, advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, df_topics, _, _ = _load(base_dir)
    if df_topics.empty:
        return []

    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    df_t = df_topics[df_topics["CommentID"].isin(df["CommentID"])].copy()
    df_t["topic_name"] = df_t["topic_name"].str.strip().str.lower().str.replace("_", " ", regex=False)
    df_t["topic_name"] = df_t["topic_name"].apply(_taxonomy_topic_name)
    df_t["slug"] = df_t["topic_name"].apply(_topic_slug)

    slug_counts = df_t.groupby("slug").size().reset_index(name="count")
    slug_to_name = (
        df_t.groupby("slug")["topic_name"]
        .apply(lambda x: _best_topic_name(x.unique().tolist()))
        .reset_index()
        .rename(columns={"topic_name": "display_name"})
    )
    counts = (
        slug_counts.merge(slug_to_name, on="slug")
        .rename(columns={"display_name": "topic_name"})[["topic_name", "count"]]
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return counts.to_dict(orient="records")


def get_emerging_topics(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                        advisor_filter: str = "all", min_count: int = 3, growth_threshold: float = 0.5,
                        brand_id: str = "all") -> list:
    """Topics tÄƒng trÆ°á»Ÿng >growth_threshold% hoáº·c xuáº¥t hiá»‡n láº§n Ä‘áº§u so vá»›i ká»³ trÆ°á»›c cÃ¹ng Ä‘á»™ dÃ i."""
    df_raw, df_labels, df_topics, _, _ = _load(base_dir)

    df_curr = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df_curr.empty or df_topics.empty:
        return []

    start = pd.Timestamp(start_date)
    delta = pd.Timestamp(end_date) - start
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - delta
    df_prev = _joined(df_raw, df_labels, prev_start.strftime("%Y-%m-%d"), prev_end.strftime("%Y-%m-%d"),
                      group_id, advisor_filter, brand_id, base_dir)

    def topic_counts(df_slice):
        if df_slice.empty:
            return {}
        ids = set(df_slice["CommentID"].astype(str))
        df_t = df_topics[df_topics["CommentID"].astype(str).isin(ids)].copy()
        df_t["topic_name"] = df_t["topic_name"].str.strip().str.lower().str.replace("_", " ", regex=False)
        df_t["topic_name"] = df_t["topic_name"].apply(_taxonomy_topic_name)
        df_t["slug"] = df_t["topic_name"].apply(_topic_slug)
        slug_name = (
            df_t.groupby("slug")["topic_name"]
            .apply(lambda x: _best_topic_name(x.unique().tolist()))
            .to_dict()
        )
        counts = df_t.groupby("slug").size().to_dict()
        return {slug_name[slug]: cnt for slug, cnt in counts.items()}

    curr_counts = topic_counts(df_curr)
    prev_counts = topic_counts(df_prev)

    result = []
    for topic, curr_n in curr_counts.items():
        if curr_n < min_count:
            continue
        prev_n = prev_counts.get(topic, 0)
        if prev_n == 0:
            result.append({"topic_name": topic, "count": curr_n, "prev_count": 0, "growth_pct": None, "is_new": True})
        else:
            growth = (curr_n - prev_n) / prev_n
            if growth >= growth_threshold:
                result.append({"topic_name": topic, "count": curr_n, "prev_count": prev_n,
                                "growth_pct": round(growth * 100, 1), "is_new": False})

    result.sort(key=lambda x: (not x["is_new"], -(x["growth_pct"] or 999)))
    return result


def get_intent_distribution(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                            advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    return _intent_counts(df)


def get_crisis_alerts(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                      advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, _, df_crisis, _ = _load(base_dir)
    if df_crisis.empty:
        return []

    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    df_c = df_crisis[df_crisis["CommentID"].isin(df["CommentID"])].copy()
    df_c = df_c.merge(df_raw[["CommentID", "Content", "Date", "PostURL", "group_id"]], on="CommentID", how="left")
    df_c = df_c.merge(df_labels[["CommentID", "summary", "sentiment_score"]], on="CommentID", how="left")
    df_c = df_c.sort_values("detected_at", ascending=False)

    return _to_records(df_c)


def get_brand_mentions(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                       advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)

    mentions_path = f"{base_dir}/brand_mentions.csv"
    brands_path   = f"{base_dir}/brands.csv"
    if not os.path.exists(mentions_path) or not os.path.exists(brands_path):
        return []

    df_mentions = pd.read_csv(mentions_path)
    df_brands_raw = pd.read_csv(brands_path)
    color_col = ["color"] if "color" in df_brands_raw.columns else []
    df_brands = df_brands_raw[["brand_id", "brand_name"] + color_col]

    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty or df_mentions.empty:
        return []

    # Filter mentions to comments inside the selected period/group
    df_m = df_mentions[df_mentions["CommentID"].astype(str).isin(df["CommentID"].astype(str))].copy()
    df_m = df_m.merge(df[["CommentID", "sentiment", "crisis_level", "PostID"]], on="CommentID", how="left")
    df_m = df_m.merge(df_brands, on="brand_id", how="left")
    df_m["PostID"] = df_m["PostID"].astype(str)

    total_all = len(df_m)

    # â”€â”€ Engagement: avg replies per post that mentions each brand â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Replies = comments in the period where ReplyToID is non-empty
    df_r = df_raw.copy()
    df_r["Date"] = _parse_comment_dates(df_r["Date"])
    df_r = df_r[(df_r["Date"] >= pd.Timestamp(start_date)) & (df_r["Date"] <= pd.Timestamp(end_date))]
    df_r["ReplyToID"] = df_r["ReplyToID"].fillna("").astype(str).str.strip()
    df_r["PostID"] = df_r["PostID"].astype(str)
    replies_per_post = df_r[df_r["ReplyToID"] != ""].groupby("PostID").size().to_dict()

    # â”€â”€ Trend: SoV delta vs previous equal-length period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    period_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 1)
    prev_end_dt   = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    prev_start_dt = prev_end_dt - pd.Timedelta(days=period_days - 1)
    df_prev = _joined(df_raw, df_labels, prev_start_dt.strftime("%Y-%m-%d"), prev_end_dt.strftime("%Y-%m-%d"),
                      group_id, advisor_filter, brand_id, base_dir)
    if not df_prev.empty:
        df_m_prev = df_mentions[df_mentions["CommentID"].astype(str).isin(df_prev["CommentID"].astype(str))]
        total_prev = len(df_m_prev)
        prev_sov = (df_m_prev.groupby("brand_id").size() / total_prev * 100).to_dict() if total_prev > 0 else {}
    else:
        prev_sov = {}

    result = []
    for brand_id, grp in df_m.groupby("brand_id"):
        total    = len(grp)
        positive = int((grp["sentiment"] == "tich_cuc").sum())
        negative = int((grp["sentiment"] == "tieu_cuc").sum())
        neutral  = int((grp["sentiment"] == "trung_lap").sum())
        name  = grp["brand_name"].dropna().iloc[0] if not grp["brand_name"].dropna().empty else brand_id
        color = grp["color"].dropna().iloc[0] if "color" in grp.columns and not grp["color"].dropna().empty else None

        # Crisis-free %: mentions where crisis_level is none/null
        crisis_vals = grp["crisis_level"].fillna("none").str.strip().str.lower()
        crisis_free_pct = round((crisis_vals == "none").sum() / total * 100, 1)

        # SoV %
        sov_pct = round(total / total_all * 100, 1) if total_all > 0 else 0.0

        # Engagement: avg replies across posts that contain this brand's mentions
        brand_post_ids = grp["PostID"].dropna().unique()
        if len(brand_post_ids) > 0:
            avg_replies = sum(replies_per_post.get(pid, 0) for pid in brand_post_ids) / len(brand_post_ids)
        else:
            avg_replies = 0.0

        # SoV delta vs prev period (positive = growing)
        sov_delta = round(sov_pct - prev_sov.get(brand_id, sov_pct), 1)

        result.append({
            "brand_id":        brand_id,
            "brand_name":      str(name),
            "color":           color,
            "mentions":        total,
            "positive_pct":    round(positive / total * 100, 1),
            "negative_pct":    round(negative / total * 100, 1),
            "neutral_pct":     round(neutral  / total * 100, 1),
            "sov_pct":         sov_pct,
            "crisis_free_pct": crisis_free_pct,
            "_avg_replies":    avg_replies,   # temp, normalized below
            "sov_delta":       sov_delta,
        })

    # Normalize engagement to 0â€“100 relative to the highest brand this period
    max_eng = max((r["_avg_replies"] for r in result), default=1) or 1
    for r in result:
        r["engagement_score"] = round(r["_avg_replies"] / max_eng * 100, 1)
        del r["_avg_replies"]

    return sorted(result, key=lambda x: x["mentions"], reverse=True)


def get_pain_clusters(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                      advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    cluster_map = _pain_cluster_map(base_dir)

    rows = []
    for _, row in df.iterrows():
        try:
            points = json.loads(row["pain_points"]) if pd.notna(row.get("pain_points")) else []
        except Exception:
            points = []
        for pt in points:
            pt_c = str(pt).strip().lower()
            if pt_c:
                topic_id = _taxonomy_negative_topic(pt_c, row.get("Content", ""), row.get("summary", ""))
                rows.append({
                    "CommentID": row["CommentID"],
                    "cluster_name": cluster_map.get(pt_c, NEGATIVE_TOPIC_LABELS.get(topic_id, "Khac")),
                    "pain_point": pt_c,
                })

    if not rows:
        return []

    df_pp = pd.DataFrame(rows).drop_duplicates(subset=["CommentID", "cluster_name"])

    result = []
    for cluster, grp in df_pp.groupby("cluster_name"):
        examples = grp["pain_point"].value_counts().head(3).index.tolist()
        result.append({"cluster_name": cluster, "count": len(grp), "examples": examples})

    result.sort(key=lambda x: (x["cluster_name"] == "Khac", -x["count"]))
    return result


def get_pain_cluster_trend(start_date: str, end_date: str, group_id: str, granularity: str = "week",
                           base_dir: str = "./data", advisor_filter: str = "all",
                           top_n: int = 5, brand_id: str = "all") -> dict:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return {"clusters": [], "data": []}

    cluster_map = _pain_cluster_map(base_dir)

    rows = []
    for _, row in df.iterrows():
        try:
            points = json.loads(row["pain_points"]) if pd.notna(row.get("pain_points")) else []
        except Exception:
            points = []
        for pt in points:
            pt_c = str(pt).strip().lower()
            topic_id = _taxonomy_negative_topic(pt_c, row.get("Content", ""), row.get("summary", ""))
            cluster = cluster_map.get(pt_c, NEGATIVE_TOPIC_LABELS.get(topic_id, "Khac"))
            if pt_c and cluster != "Khac":
                rows.append({"Date": row["Date"], "cluster_name": cluster, "CommentID": row["CommentID"]})

    if not rows:
        return {"clusters": [], "data": []}

    df_pp = pd.DataFrame(rows).drop_duplicates(subset=["CommentID", "cluster_name"])
    top_clusters = df_pp["cluster_name"].value_counts().head(top_n).index.tolist()
    df_pp = df_pp[df_pp["cluster_name"].isin(top_clusters)]

    freq = {"day": "D", "week": "W", "month": "M", "year": "Y"}.get(granularity, "W")
    df_pp["period"] = df_pp["Date"].dt.to_period(freq).apply(lambda r: r.start_time.strftime("%Y-%m-%d"))

    pivot = df_pp.groupby(["period", "cluster_name"]).size().unstack(fill_value=0)
    for c in top_clusters:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[top_clusters].reset_index()

    data = []
    for _, row in pivot.iterrows():
        entry = {"period": row["period"]}
        for c in top_clusters:
            entry[c] = int(row[c])
        data.append(entry)

    return {"clusters": top_clusters, "data": sorted(data, key=lambda x: x["period"])}


def get_thread_analysis(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                        top_n: int = 15, advisor_filter: str = "all", brand_id: str = "all") -> list:
    df_raw, df_labels, df_topics, _, _ = _load(base_dir)
    if df_raw.empty:
        return []

    # Prep raw: filter by date + group, normalize types
    df_r = df_raw.copy()
    df_r["Date"] = _parse_comment_dates(df_r["Date"])
    df_r["CommentID"] = df_r["CommentID"].astype(str)
    df_r["ReplyToID"] = df_r["ReplyToID"].fillna("").astype(str).str.strip()

    df_r = df_r[
        (df_r["Date"] >= pd.Timestamp(start_date)) &
        (df_r["Date"] <= pd.Timestamp(end_date))
    ]
    if group_id and group_id != "all":
        df_r = df_r[df_r["group_id"] == group_id]
    if brand_id and brand_id != "all":
        mentions_path = f"{base_dir}/brand_mentions.csv"
        if os.path.exists(mentions_path):
            df_mentions = pd.read_csv(mentions_path)
            brand_ids = set(
                df_mentions[df_mentions["brand_id"].astype(str) == str(brand_id)]["CommentID"].astype(str)
            )
            df_r = df_r[df_r["CommentID"].astype(str).isin(brand_ids)]
        else:
            df_r = df_r.iloc[0:0]
    if df_r.empty:
        return []

    # Labeled slice (respects exclude_advisor)
    df_lab = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if not df_lab.empty:
        df_lab["CommentID"] = df_lab["CommentID"].astype(str)
        df_lab["ReplyToID"] = df_lab["ReplyToID"].fillna("").astype(str).str.strip()

    # Advisor IDs to filter roots based on advisor_filter
    advisor_ids: set = set()
    if advisor_filter != "all" and not df_labels.empty and "is_advisor" in df_labels.columns:
        df_lk2 = df_labels.copy()
        df_lk2["CommentID"] = df_lk2["CommentID"].astype(str)
        if advisor_filter == "exclude":
            advisor_ids = set(df_lk2[df_lk2["is_advisor"] == True]["CommentID"].tolist())
        elif advisor_filter == "only":
            # keep only advisor roots â€” exclude non-advisors
            non_advisor_ids = set(df_lk2[df_lk2["is_advisor"] != True]["CommentID"].tolist())
            advisor_ids = non_advisor_ids

    # Labels lookup: CommentID -> {summary, sentiment}
    label_lookup: dict = {}
    if not df_labels.empty:
        df_lk = df_labels.copy()
        df_lk["CommentID"] = df_lk["CommentID"].astype(str)
        for _, row in df_lk[["CommentID", "summary", "sentiment"]].iterrows():
            label_lookup[row["CommentID"]] = {
                "summary": row["summary"] if pd.notna(row["summary"]) else None,
                "sentiment": row["sentiment"] if pd.notna(row["sentiment"]) else None,
            }

    # Topics lookup: CommentID -> [topic_name]
    topics_lookup: dict = {}
    if not df_topics.empty:
        df_tp = df_topics.copy()
        df_tp["CommentID"] = df_tp["CommentID"].astype(str)
        df_tp["topic_name"] = df_tp["topic_name"].str.strip().str.lower().str.replace("_", " ", regex=False)
        for cid, grp in df_tp.groupby("CommentID"):
            topics_lookup[cid] = grp["topic_name"].tolist()

    # Root comments: ReplyToID == ""
    roots = df_r[df_r["ReplyToID"] == ""]
    roots = roots[~roots["CommentID"].isin(advisor_ids)]

    # Build reply index: root_id -> list of reply rows
    reply_df = df_r[df_r["ReplyToID"] != ""]
    reply_index: dict = {}
    for _, row in reply_df.iterrows():
        reply_index.setdefault(row["ReplyToID"], []).append(row)

    result = []
    for _, root in roots.iterrows():
        root_id = root["CommentID"]
        replies = reply_index.get(root_id, [])
        reply_count = len(replies)
        if reply_count == 0:
            continue

        # Sentiment from labeled replies (exclude_advisor already applied via df_lab)
        reply_ids = [str(r["CommentID"]) for r in replies]
        labeled_replies = df_lab[df_lab["CommentID"].isin(reply_ids)] if not df_lab.empty else pd.DataFrame()
        total_lab = len(labeled_replies)
        pos = int((labeled_replies["sentiment"] == "tich_cuc").sum()) if total_lab else 0
        neg = int((labeled_replies["sentiment"] == "tieu_cuc").sum()) if total_lab else 0
        neu = int((labeled_replies["sentiment"] == "trung_lap").sum()) if total_lab else 0

        # Top 3 topics across entire thread
        thread_ids = [root_id] + reply_ids
        all_topics: list = []
        for tid in thread_ids:
            all_topics.extend(topics_lookup.get(tid, []))
        top_topics: list = []
        if all_topics:
            s = pd.Series(all_topics)
            top_topics = s.value_counts().head(3).index.tolist()

        lbl = label_lookup.get(root_id, {})
        result.append({
            "root_id":             root_id,
            "date":                root["Date"].strftime("%Y-%m-%d") if pd.notna(root["Date"]) else "",
            "group_id":            root.get("group_id", ""),
            "content":             str(root.get("Content", ""))[:200],
            "summary":             lbl.get("summary"),
            "root_sentiment":      lbl.get("sentiment"),
            "reply_count":         reply_count,
            "labeled_reply_count": total_lab,
            "positive_pct":        round(pos / total_lab * 100, 1) if total_lab else None,
            "negative_pct":        round(neg / total_lab * 100, 1) if total_lab else None,
            "neutral_pct":         round(neu / total_lab * 100, 1) if total_lab else None,
            "top_topics":          top_topics,
            "post_url":            str(root.get("PostURL", "")) if pd.notna(root.get("PostURL", None)) else "",
        })

    return sorted(result, key=lambda x: x["reply_count"], reverse=True)[:top_n]


def get_comments(start_date: str, end_date: str, group_id: str, base_dir: str = "./data",
                 advisor_filter: str = "all", limit: int = 100, brand_id: str = "all") -> list:
    df_raw, df_labels, _, _, _ = _load(base_dir)
    df = _joined(df_raw, df_labels, start_date, end_date, group_id, advisor_filter, brand_id, base_dir)
    if df.empty:
        return []

    cols = ["CommentID", "Date", "Content", "group_id", "PostURL",
            "intent", "sentiment", "sentiment_score", "summary", "is_advisor", "is_seeding"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["is_advisor"] = df["is_advisor"].fillna(False).astype(bool)
    df = df.sort_values("Date", ascending=False).head(limit)
    return _to_records(df)


def get_groups(base_dir: str = "./data") -> list:
    path = f"{base_dir}/groups.csv"
    if not os.path.exists(path):
        return []
    return _to_records(pd.read_csv(path))


def get_brands(base_dir: str = "./data") -> list:
    path = f"{base_dir}/brands.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    cols = [c for c in ["brand_id", "brand_name"] if c in df.columns]
    return _to_records(df[cols].dropna(subset=["brand_id", "brand_name"]).sort_values("brand_name"))


def get_campaign_impact(base_dir: str = "./data", start_date: str | None = None, end_date: str | None = None,
                        group_id: str = "all", advisor_filter: str = "all", granularity: str = "week",
                        brand_id: str = "all", insurance_lines: str = "all") -> dict:
    campaigns_path = f"{base_dir}/campaigns.csv"
    mentions_path = f"{base_dir}/brand_mentions.csv"
    if not os.path.exists(campaigns_path):
        return {"available": False, "campaigns": [], "message": "data/campaigns.csv not found."}

    df_campaigns = pd.read_csv(campaigns_path).dropna(subset=["campaign_name", "brand_id", "start_date", "end_date"])
    if df_campaigns.empty:
        return {"available": False, "campaigns": [], "message": "No campaigns configured."}
    if not os.path.exists(mentions_path):
        return {"available": False, "campaigns": [], "message": "brand_mentions.csv not found."}

    _, _, df_topics, _, _ = _load(base_dir)
    df_mentions = pd.read_csv(mentions_path)
    df_all = _joined_all(base_dir)
    if df_all.empty:
        return {"available": True, "campaigns": [], "total_timeline": [], "topic_shift": []}
    df_all["CommentID"] = df_all["CommentID"].astype(str)
    if group_id and group_id != "all":
        df_all = df_all[df_all["group_id"].astype(str) == str(group_id)]
    if "is_advisor" in df_all.columns:
        if advisor_filter == "exclude":
            df_all = df_all[df_all["is_advisor"] != True]
        elif advisor_filter == "only":
            df_all = df_all[df_all["is_advisor"] == True]
    if brand_id and brand_id != "all":
        ids = set(df_mentions[df_mentions["brand_id"].astype(str) == str(brand_id)]["CommentID"].astype(str))
        df_all = df_all[df_all["CommentID"].isin(ids)]
    df_all = _filter_insurance_lines(df_all, base_dir, insurance_lines)
    if start_date and end_date:
        df = df_all[(df_all["Date"] >= pd.Timestamp(start_date)) & (df_all["Date"] <= pd.Timestamp(end_date))].copy()
    else:
        df = df_all.copy()
    df["CommentID"] = df["CommentID"].astype(str)
    df_mentions["CommentID"] = df_mentions["CommentID"].astype(str)
    planning_path = f"{base_dir}/planning_insights.csv"
    df_planning = _read_csv_cached(planning_path) if os.path.exists(planning_path) else pd.DataFrame()
    if not df_planning.empty and "CommentID" in df_planning:
        df_planning["CommentID"] = df_planning["CommentID"].astype(str)

    timeline_df = _periodize(df, granularity) if start_date and end_date else pd.DataFrame()
    total_timeline = []
    if not timeline_df.empty:
        for period, grp in timeline_df.groupby("period"):
            total_timeline.append({
                "period": period,
                "positive_pct": float(round((grp["sentiment"] == "tich_cuc").mean() * 100, 1)),
                "mentions": int(len(grp)),
            })

    results = []
    topic_shift_rows = []
    for _, cp in df_campaigns.iterrows():
        cp_brand_id = str(cp["brand_id"])
        start = pd.Timestamp(cp["start_date"])
        end = pd.Timestamp(cp["end_date"])
        duration = max((end - start).days + 1, 1)
        pre_start = start - pd.Timedelta(days=duration)
        pre_end = start - pd.Timedelta(days=1)
        post_start = end + pd.Timedelta(days=1)
        post_end = end + pd.Timedelta(days=duration)

        brand_comment_ids = set(df_mentions[df_mentions["brand_id"].astype(str) == cp_brand_id]["CommentID"])
        df_brand = df_all[df_all["CommentID"].isin(brand_comment_ids)].copy()
        pre = df_brand[(df_brand["Date"] >= pre_start) & (df_brand["Date"] <= pre_end)]
        during = df_brand[(df_brand["Date"] >= start) & (df_brand["Date"] <= end)]
        post = df_brand[(df_brand["Date"] >= post_start) & (df_brand["Date"] <= post_end)]

        def positive_pct(slice_df):
            return float(round((slice_df["sentiment"] == "tich_cuc").mean() * 100, 1)) if len(slice_df) else 0.0

        def negative_pct(slice_df):
            return float(round((slice_df["sentiment"] == "tieu_cuc").mean() * 100, 1)) if len(slice_df) else 0.0

        def avg_score(slice_df):
            return float(round(pd.to_numeric(slice_df["sentiment_score"], errors="coerce").mean(), 3)) if len(slice_df) else 0.0

        def intent_count(slice_df, intents):
            return int(slice_df["intent"].isin(intents).sum()) if len(slice_df) and "intent" in slice_df else 0

        pre_pos = positive_pct(pre)
        during_pos = positive_pct(during)
        post_pos = positive_pct(post)
        pre_mentions = len(pre)
        during_mentions = len(during)
        post_mentions = len(post)
        mention_lift = round((during_mentions - pre_mentions) / pre_mentions * 100, 1) if pre_mentions else None
        sent_lift = float(round(during_pos - pre_pos, 1))
        negative_lift = float(round(negative_pct(during) - negative_pct(pre), 1))
        score_lift = float(round(avg_score(during) - avg_score(pre), 3))
        intent_pre = intent_count(pre, ["hoi_gia", "so_sanh"])
        intent_during = intent_count(during, ["hoi_gia", "so_sanh"])
        intent_lift = round((intent_during - intent_pre) / intent_pre * 100, 1) if intent_pre else None

        def planning_slice(slice_df):
            if df_planning.empty:
                return pd.DataFrame()
            return df_planning[df_planning["CommentID"].isin(set(slice_df["CommentID"].astype(str)))]

        pre_plan = planning_slice(pre)
        during_plan = planning_slice(during)

        def dist_delta(column, limit=8):
            pre_dist = {r["label"]: r["count"] for r in _planning_distribution(pre_plan, column, limit=50)}
            dur_dist = {r["label"]: r["count"] for r in _planning_distribution(during_plan, column, limit=50)}
            labels = sorted(set(pre_dist) | set(dur_dist), key=lambda k: dur_dist.get(k, 0), reverse=True)[:limit]
            return [{"label": k, "pre": int(pre_dist.get(k, 0)), "during": int(dur_dist.get(k, 0)),
                     "delta": int(dur_dist.get(k, 0) - pre_dist.get(k, 0))} for k in labels]

        def score_delta(column, limit=8):
            pre_scores = {r["label"]: r["score"] for r in _planning_score_average(pre_plan, column, limit=50)}
            dur_scores = {r["label"]: r["score"] for r in _planning_score_average(during_plan, column, limit=50)}
            labels = sorted(set(pre_scores) | set(dur_scores), key=lambda k: dur_scores.get(k, 0), reverse=True)[:limit]
            return [{"label": k, "pre": float(pre_scores.get(k, 0)), "during": float(dur_scores.get(k, 0)),
                     "delta": float(round(dur_scores.get(k, 0) - pre_scores.get(k, 0), 3))} for k in labels]

        campaign_keywords = []
        for col in ("topic_keywords", "message_keywords", "main_claim"):
            raw_keywords = str(cp.get(col, "") or "")
            campaign_keywords.extend([k.strip().lower() for k in raw_keywords.split("|") if k.strip()])
        campaign_keywords = list(dict.fromkeys(campaign_keywords))
        matched = pd.DataFrame()
        if campaign_keywords and not during.empty:
            content_l = during["Content"].fillna("").astype(str).str.lower()
            mask = content_l.apply(lambda text: any(k in text for k in campaign_keywords))
            matched = during[mask].copy()
        resonance = {
            "keywords": campaign_keywords,
            "matched_count": int(len(matched)),
            "matched_positive_pct": positive_pct(matched),
            "matched_negative_pct": negative_pct(matched),
            "sample_comments": _to_records(matched.sort_values("Date", ascending=False)[
                [c for c in ["CommentID", "Date", "Content", "sentiment", "summary"] if c in matched.columns]
            ].head(12)) if not matched.empty else [],
        }
        barrier_scores = score_delta("barrier_scores")
        trigger_scores = score_delta("trigger_scores")
        brand_impact_score = float(max(0, min(100, round(
            50 + sent_lift * 1.5 - max(0, negative_lift) * 1.2 + (mention_lift or 0) * 0.05 + (intent_lift or 0) * 0.05,
            1,
        ))))
        keyword_share = (len(matched) / during_mentions * 100) if during_mentions else 0.0
        message_sentiment_bonus = resonance["matched_positive_pct"] - max(0, resonance["matched_negative_pct"] * 0.7)
        message_resonance_score = float(max(0, min(100, round(
            min(50, keyword_share * 2.0) + max(0, message_sentiment_bonus) * 0.5 + min(20, len(campaign_keywords) * 2),
            1,
        ))))
        overall_campaign_score = float(round(brand_impact_score * 0.65 + message_resonance_score * 0.35, 1))
        resonance["keyword_share_of_brand_mentions"] = float(round(keyword_share, 2))
        resonance["message_resonance_score"] = message_resonance_score

        insight_cards = []
        if during_mentions > pre_mentions:
            insight_cards.append({
                "title": "Awareness increased",
                "severity": "positive",
                "summary": f"Brand mentions moved from {pre_mentions} pre-campaign to {during_mentions} during campaign.",
                "action": "Check whether the lift is accompanied by decision intent and lower barriers.",
            })
        if sent_lift > 5:
            insight_cards.append({
                "title": "Sentiment improved",
                "severity": "positive",
                "summary": f"Positive sentiment increased by {sent_lift} percentage points during campaign.",
                "action": "Reuse the messages and channels that created positive evidence.",
            })
        if negative_lift > 5:
            insight_cards.append({
                "title": "Risk also increased",
                "severity": "warning",
                "summary": f"Negative sentiment increased by {negative_lift} percentage points during campaign.",
                "action": "Review top barriers and prepare response content before scaling.",
            })
        top_barrier = barrier_scores[0] if barrier_scores else None
        if top_barrier and top_barrier["during"] >= 0.35:
            insight_cards.append({
                "title": "Barrier remains material",
                "severity": "warning",
                "summary": f"Top barrier signal is {top_barrier['label']} with score {top_barrier['during']}.",
                "action": "Add creative that directly reduces this barrier with proof and concrete examples.",
            })
        if campaign_keywords and not len(matched):
            insight_cards.append({
                "title": "Message resonance is weak",
                "severity": "warning",
                "summary": "No during-campaign comments matched configured campaign message keywords.",
                "action": "Review campaign keywords/main claim or add more representative message terms.",
            })

        results.append({
            "campaign_name": cp["campaign_name"],
            "brand_id": cp_brand_id,
            "objective": cp.get("campaign_objective", ""),
            "target_audience": cp.get("target_audience", ""),
            "main_claim": cp.get("main_claim", ""),
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "pre_start": pre_start.strftime("%Y-%m-%d"),
            "pre_end": pre_end.strftime("%Y-%m-%d"),
            "post_start": post_start.strftime("%Y-%m-%d"),
            "post_end": post_end.strftime("%Y-%m-%d"),
            "pre_positive_pct": pre_pos,
            "during_positive_pct": during_pos,
            "post_positive_pct": post_pos,
            "pre_negative_pct": negative_pct(pre),
            "during_negative_pct": negative_pct(during),
            "post_negative_pct": negative_pct(post),
            "pre_avg_score": avg_score(pre),
            "during_avg_score": avg_score(during),
            "post_avg_score": avg_score(post),
            "sentiment_lift": sent_lift,
            "negative_lift": negative_lift,
            "score_lift": score_lift,
            "pre_mentions": pre_mentions,
            "during_mentions": during_mentions,
            "post_mentions": post_mentions,
            "mention_lift_pct": mention_lift,
            "intent_pre": intent_pre,
            "intent_during": intent_during,
            "intent_lift_pct": intent_lift,
            "score": overall_campaign_score,
            "overall_campaign_score": overall_campaign_score,
            "brand_impact_score": brand_impact_score,
            "message_resonance_score": message_resonance_score,
            "planning_coverage": int(len(during_plan)),
            "decision_stage_shift": dist_delta("decision_stage"),
            "brand_personality_shift": dist_delta("brand_personality"),
            "brand_tension_shift": dist_delta("brand_tension"),
            "latent_barrier_shift": dist_delta("latent_barrier"),
            "purchase_trigger_shift": dist_delta("purchase_trigger"),
            "barrier_scores": barrier_scores,
            "trigger_scores": trigger_scores,
            "message_resonance": resonance,
            "insight_cards": insight_cards,
            "visible": bool(
                start_date and end_date
                and start <= pd.Timestamp(end_date)
                and end >= pd.Timestamp(start_date)
                and (brand_id == "all" or brand_id == cp_brand_id)
            ),
            "visible_start": max(start, pd.Timestamp(start_date)).strftime("%Y-%m-%d") if start_date and end_date and start <= pd.Timestamp(end_date) and end >= pd.Timestamp(start_date) else None,
            "visible_end": min(end, pd.Timestamp(end_date)).strftime("%Y-%m-%d") if start_date and end_date and start <= pd.Timestamp(end_date) and end >= pd.Timestamp(start_date) else None,
        })

        keywords = str(cp.get("topic_keywords", "") or "")
        if keywords and not df_topics.empty:
            kws = [k.strip().lower() for k in keywords.split("|") if k.strip()]
            df_topic = df_topics.copy()
            df_topic["CommentID"] = df_topic["CommentID"].astype(str)
            df_topic["topic_name"] = df_topic["topic_name"].astype(str).str.lower()
            pre_ids = set(pre["CommentID"])
            post_ids = set(during["CommentID"])
            for kw in kws:
                pre_count = int(df_topic[df_topic["CommentID"].isin(pre_ids) & df_topic["topic_name"].str.contains(kw, regex=False)].shape[0])
                post_count = int(df_topic[df_topic["CommentID"].isin(post_ids) & df_topic["topic_name"].str.contains(kw, regex=False)].shape[0])
                lift = round((post_count - pre_count) / pre_count * 100, 1) if pre_count else None
                topic_shift_rows.append({
                    "campaign_name": cp["campaign_name"],
                    "topic": kw,
                    "pre_count": pre_count,
                    "post_count": post_count,
                    "lift_pct": lift,
                    "is_new": pre_count == 0 and post_count > 0,
                })

    return _json_safe({
        "available": True,
        "campaigns": results,
        "total_timeline": sorted(total_timeline, key=lambda x: x["period"]),
        "topic_shift": topic_shift_rows,
    })


# Spec-aligned dashboard helpers
def _split_filter(value: str | None) -> list[str]:
    if not value or value == "all":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip() and v.strip() != "all"]


def _spec_joined(base_dir: str, start_date: str, end_date: str, groups: str = "all",
                 brands: str = "all", sentiments: str = "all", advisor_filter: str = "all",
                 insurance_lines: str = "all") -> pd.DataFrame:
    df = _joined_all(base_dir)
    if df.empty:
        return df
    df = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]
    if "is_advisor" in df.columns:
        if advisor_filter == "exclude":
            df = df[df["is_advisor"] != True]
        elif advisor_filter == "only":
            df = df[df["is_advisor"] == True]
    group_values = _split_filter(groups)
    if group_values:
        df = df[df["group_id"].astype(str).isin(group_values)]
    brand_values = _split_filter(brands)
    if brand_values:
        mentions_path = f"{base_dir}/brand_mentions.csv"
        if not os.path.exists(mentions_path):
            return df.iloc[0:0]
        df_mentions = pd.read_csv(mentions_path)
        ids = set(df_mentions[df_mentions["brand_id"].astype(str).isin(brand_values)]["CommentID"].astype(str))
        df = df[df["CommentID"].astype(str).isin(ids)]
    sentiment_values = _split_filter(sentiments)
    if sentiment_values:
        df = df[df["sentiment"].astype(str).isin(sentiment_values)]
    df = _filter_insurance_lines(df, base_dir, insurance_lines)
    return df


def _spec_context(base_dir: str, start: str, end: str, compare_start: str | None,
                  compare_end: str | None, groups: str, brands: str, sentiments: str,
                  advisor_filter: str, insurance_lines: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    curr = _spec_joined(base_dir, start, end, groups, brands, sentiments, advisor_filter, insurance_lines)
    if not compare_start or not compare_end:
        compare_start, compare_end = _previous_period(start, end)
    prev = _spec_joined(base_dir, compare_start, compare_end, groups, brands, sentiments, advisor_filter, insurance_lines)
    return curr, prev


def _pain_cluster_map(base_dir: str) -> dict:
    clusters_path = f"{base_dir}/pain_clusters.csv"
    if not os.path.exists(clusters_path):
        return {}
    clusters_df = _read_csv_cached(clusters_path)
    if clusters_df.empty or "pain_point" not in clusters_df or "cluster_name" not in clusters_df:
        return {}
    return dict(zip(clusters_df["pain_point"].astype(str).str.strip().str.lower(), clusters_df["cluster_name"]))


def _pain_rows(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    if df.empty or "pain_points" not in df:
        return pd.DataFrame()
    cluster_map = _pain_cluster_map(base_dir)
    rows = []
    keep_cols = ["CommentID", "Date", "Content", "sentiment", "sentiment_score", "intent", "summary", "group_id"]
    for _, row in df.iterrows():
        try:
            points = json.loads(row["pain_points"]) if pd.notna(row.get("pain_points")) else []
        except Exception:
            points = []
        for pt in points:
            pt_c = str(pt).strip().lower()
            if not pt_c:
                continue
            topic_id = _taxonomy_negative_topic(pt_c, row.get("Content", ""), row.get("summary", ""))
            cluster_name = cluster_map.get(pt_c, NEGATIVE_TOPIC_LABELS.get(topic_id, "Khac"))
            details = _advisor_details(pt_c, row.get("Content", ""), row.get("summary", "")) if topic_id == "dich_vu_tu_van" else []
            item = {col: row.get(col) for col in keep_cols if col in df.columns}
            item.update({
                "pain_point": pt_c,
                "cluster_name": cluster_name,
                "advisor_details": details,
                "advisor_detail_labels": [ADVISOR_DETAIL_LABELS[d] for d in details],
            })
            rows.append(item)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).drop_duplicates(subset=["CommentID", "cluster_name"])
    out["sentiment_score"] = pd.to_numeric(out.get("sentiment_score"), errors="coerce")
    return out


def _brand_lookup(base_dir: str) -> dict:
    brands_path = f"{base_dir}/brands.csv"
    if not os.path.exists(brands_path):
        return {}
    df_brands = _read_csv_cached(brands_path)
    if df_brands.empty or "brand_id" not in df_brands:
        return {}
    name_col = "brand_name" if "brand_name" in df_brands else "brand_id"
    return dict(zip(df_brands["brand_id"].astype(str), df_brands[name_col].astype(str)))


def _comment_brand_names(base_dir: str, comment_ids: set[str]) -> pd.DataFrame:
    mentions_path = f"{base_dir}/brand_mentions.csv"
    if not os.path.exists(mentions_path) or not comment_ids:
        return pd.DataFrame(columns=["CommentID", "brand_id", "brand_name"])
    mentions = _read_csv_cached(mentions_path)
    if mentions.empty or "CommentID" not in mentions or "brand_id" not in mentions:
        return pd.DataFrame(columns=["CommentID", "brand_id", "brand_name"])
    mentions["CommentID"] = mentions["CommentID"].astype(str)
    mentions = mentions[mentions["CommentID"].isin(comment_ids)].copy()
    names = _brand_lookup(base_dir)
    mentions["brand_name"] = mentions["brand_id"].astype(str).map(names).fillna(mentions["brand_id"].astype(str))
    return mentions[["CommentID", "brand_id", "brand_name"]]


def _decode_multi_cell(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v)]
    except Exception:
        pass
    return [v.strip() for v in text.split("|") if v.strip()]


def _decode_score_cell(value) -> dict:
    if pd.isna(value) or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _planning_distribution(df: pd.DataFrame, column: str, limit: int = 12) -> list[dict]:
    counts: dict[str, int] = {}
    if df.empty or column not in df:
        return []
    for value in df[column]:
        items = _decode_multi_cell(value) if str(value).strip().startswith("[") else [str(value).strip()]
        for item in items:
            if item and item not in ("unknown", "none_unknown", "nan", "None"):
                counts[item] = counts.get(item, 0) + 1
    return [{"label": k, "count": int(v)} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]]


def _planning_score_average(df: pd.DataFrame, column: str, limit: int = 12) -> list[dict]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    if df.empty or column not in df:
        return []
    for value in df[column]:
        scores = _decode_score_cell(value)
        for key, score in scores.items():
            try:
                val = float(score)
            except Exception:
                continue
            totals[key] = totals.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
    rows = [{"label": k, "score": round(totals[k] / counts[k], 3)} for k in totals if counts.get(k)]
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:limit]


def _planning_summary(base_dir: str, comment_ids: set[str]) -> dict:
    path = f"{base_dir}/planning_insights.csv"
    if not os.path.exists(path):
        return {"available": False, "message": "Run src.generate_planning_insights to create planning data."}
    df = _read_csv_cached(path)
    if df.empty or "CommentID" not in df:
        return {"available": False, "message": "planning_insights.csv is empty."}
    df["CommentID"] = df["CommentID"].astype(str)
    total_all = len(df)
    if comment_ids:
        df = df[df["CommentID"].isin(comment_ids)]
    total = len(df)
    if total == 0:
        return {
            "available": True,
            "total": 0,
            "total_all": int(total_all),
            "message": "Planning data exists, but no rows match the current date/group/brand/sentiment filters.",
        }
    confidence = pd.to_numeric(df.get("confidence"), errors="coerce")
    samples_cols = ["CommentID", "preferred_brand", "brand_rejection", "brand_personality",
                    "brand_tension", "decision_stage", "confidence", "evidence"]
    samples = df.sort_values("confidence", ascending=False)[[c for c in samples_cols if c in df.columns]].head(30)
    return {
        "available": True,
        "total": int(total),
        "avg_confidence": float(round(confidence.mean(), 3)) if not confidence.dropna().empty else None,
        "preferred_brand": _planning_distribution(df, "preferred_brand"),
        "brand_rejection": _planning_distribution(df, "brand_rejection"),
        "brand_personality": _planning_distribution(df, "brand_personality"),
        "brand_tension": _planning_distribution(df, "brand_tension"),
        "product_style": _planning_distribution(df, "preferred_product_style"),
        "comm_style": _planning_distribution(df, "preferred_comm_style"),
        "premium_band": _planning_distribution(df, "premium_affordability_band"),
        "premium_sensitivity": _planning_distribution(df, "premium_sensitivity"),
        "literacy": _planning_distribution(df, "insurance_literacy"),
        "terms_awareness": _planning_distribution(df, "terms_awareness"),
        "purchase_trigger": _planning_distribution(df, "purchase_trigger"),
        "referral_type": _planning_distribution(df, "referral_type"),
        "explicit_barrier": _planning_distribution(df, "explicit_barrier"),
        "latent_barrier": _planning_distribution(df, "latent_barrier"),
        "decision_stage": _planning_distribution(df, "decision_stage"),
        "trigger_scores": _planning_score_average(df, "trigger_scores"),
        "barrier_scores": _planning_score_average(df, "barrier_scores"),
        "samples": _to_records(samples),
    }


def _crisis_by_brand(df: pd.DataFrame, base_dir: str, limit: int = 10) -> list[dict]:
    if df.empty or "crisis_level" not in df:
        return []
    crisis = df[df["crisis_level"].isin(["low", "medium", "high"])].copy()
    if crisis.empty:
        return []
    crisis["CommentID"] = crisis["CommentID"].astype(str)
    mentions = _comment_brand_names(base_dir, set(crisis["CommentID"]))
    if mentions.empty:
        crisis["brand_id"] = "unknown"
        crisis["brand_name"] = "Unknown brand"
        branded = crisis[["CommentID", "brand_id", "brand_name", "crisis_level"]]
    else:
        branded = crisis[["CommentID", "crisis_level"]].merge(mentions, on="CommentID", how="left")
        branded["brand_id"] = branded["brand_id"].fillna("unknown")
        branded["brand_name"] = branded["brand_name"].fillna("Unknown brand")
    counts = (
        branded
        .groupby(["brand_id", "brand_name", "crisis_level"])
        .size()
        .reset_index(name="count")
    )
    rows = []
    for (brand_id, brand_name), grp in counts.groupby(["brand_id", "brand_name"]):
        by_level = {str(row["crisis_level"]): int(row["count"]) for _, row in grp.iterrows()}
        total = sum(by_level.values())
        rows.append({
            "brand_id": str(brand_id),
            "brand_name": str(brand_name),
            "low": by_level.get("low", 0),
            "medium": by_level.get("medium", 0),
            "high": by_level.get("high", 0),
            "total": int(total),
        })
    return sorted(rows, key=lambda r: r["total"], reverse=True)[:limit]


def get_overview_spec(start: str, end: str, compare_start: str | None = None, compare_end: str | None = None,
                      groups: str = "all", brands: str = "all", sentiments: str = "all",
                      advisor_filter: str = "all", base_dir: str = "./data",
                      insurance_lines: str = "all") -> dict:
    curr, prev = _spec_context(base_dir, start, end, compare_start, compare_end, groups, brands, sentiments, advisor_filter, insurance_lines)

    def metrics(df: pd.DataFrame) -> dict:
        total = len(df)
        if total == 0:
            return {"total": 0, "positive_pct": 0.0, "super_negative_pct": 0.0, "crisis_count": 0}
        crisis_count = int(df["crisis_level"].isin(["low", "medium", "high"]).sum()) if "crisis_level" in df else 0
        return {
            "total": total,
            "positive_pct": float(round((df["sentiment"] == "tich_cuc").mean() * 100, 1)),
            "super_negative_pct": float(round(((df["sentiment"] == "tieu_cuc") & (df["sentiment_score"] <= -0.7)).mean() * 100, 1)),
            "crisis_count": crisis_count,
        }

    c = metrics(curr)
    p = metrics(prev)
    return {
        "current": c,
        "previous": p,
        "delta": {
            "total": c["total"] - p["total"],
            "positive_pct": round(c["positive_pct"] - p["positive_pct"], 1),
            "super_negative_pct": round(c["super_negative_pct"] - p["super_negative_pct"], 1),
            "crisis_count": c["crisis_count"] - p["crisis_count"],
        },
        "crisis_by_brand": _crisis_by_brand(curr, base_dir),
    }


def get_trends_spec(start: str, end: str, compare_start: str | None = None, compare_end: str | None = None,
                    groups: str = "all", brands: str = "all", sentiments: str = "all",
                    advisor_filter: str = "all", granularity: str = "week",
                    base_dir: str = "./data", insurance_lines: str = "all") -> dict:
    curr, prev = _spec_context(base_dir, start, end, compare_start, compare_end, groups, brands, sentiments, advisor_filter, insurance_lines)
    _, _, df_topics, _, _ = _load(base_dir)
    curr_p = _periodize(curr, granularity)
    prev_p = _periodize(prev, granularity)
    sentiment = []
    for period, grp in curr_p.groupby("period"):
        sentiment.append({
            "period": period,
            "positive": int((grp["sentiment"] == "tich_cuc").sum()),
            "neutral": int((grp["sentiment"] == "trung_lap").sum()),
            "negative": int((grp["sentiment"] == "tieu_cuc").sum()),
        })

    if df_topics.empty or curr.empty:
        topic_trend, emerging = [], []
    else:
        df_topics = df_topics.copy()
        df_topics["CommentID"] = df_topics["CommentID"].astype(str)
        df_topics["topic_name"] = df_topics["topic_name"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        df_topics["topic_name"] = df_topics["topic_name"].apply(_taxonomy_topic_name)
        curr_ids = set(curr["CommentID"].astype(str))
        prev_ids = set(prev["CommentID"].astype(str))
        curr_t = df_topics[df_topics["CommentID"].isin(curr_ids)]
        prev_t = df_topics[df_topics["CommentID"].isin(prev_ids)]
        curr_counts = curr_t["topic_name"].value_counts()
        prev_counts = prev_t["topic_name"].value_counts()
        topic_trend = [
            {
                "topic": topic,
                "count": int(count),
                "previous": int(prev_counts.get(topic, 0)),
                "change": int(count) - int(prev_counts.get(topic, 0)),
            }
            for topic, count in curr_counts.head(10).items()
        ]
        emerging = []
        for topic, count in curr_counts.items():
            prev_count = int(prev_counts.get(topic, 0))
            if prev_count == 0 or (prev_count > 0 and (count - prev_count) / prev_count > 0.5):
                emerging.append({
                    "topic": topic,
                    "count": int(count),
                    "previous": prev_count,
                    "growth_pct": None if prev_count == 0 else round((count - prev_count) / prev_count * 100, 1),
                    "is_new": prev_count == 0,
                })
        emerging = sorted(emerging, key=lambda r: r["count"], reverse=True)[:50]

    intents = _intent_counts(curr)
    return {"sentiment": sorted(sentiment, key=lambda x: x["period"]), "topic_trend": topic_trend,
            "emerging": emerging, "intents": intents}


def get_brand_intelligence_spec(start: str, end: str, groups: str = "all", brands: str = "all",
                                sentiments: str = "all", advisor_filter: str = "exclude",
                                base_dir: str = "./data", insurance_lines: str = "all") -> dict:
    df = _spec_joined(base_dir, start, end, groups, brands, sentiments, advisor_filter, insurance_lines)
    _, _, df_topics, _, _ = _load(base_dir)
    mentions_path = f"{base_dir}/brand_mentions.csv"
    brands_path = f"{base_dir}/brands.csv"
    if df.empty or not os.path.exists(mentions_path) or not os.path.exists(brands_path):
        return {"brands": [], "topic_matrix": [], "agent_ratio": []}
    df_mentions = pd.read_csv(mentions_path)
    df_brands = pd.read_csv(brands_path)
    df_mentions["CommentID"] = df_mentions["CommentID"].astype(str)
    df["CommentID"] = df["CommentID"].astype(str)
    m = df_mentions[df_mentions["CommentID"].isin(set(df["CommentID"]))].merge(
        df[["CommentID", "sentiment_score", "sentiment", "is_advisor"]], on="CommentID", how="left"
    ).merge(df_brands[["brand_id", "brand_name"]], on="brand_id", how="left")
    total_mentions = len(m) or 1
    rows = []
    for bid, grp in m.groupby("brand_id"):
        rows.append({
            "brand_id": bid,
            "brand_name": grp["brand_name"].dropna().iloc[0] if not grp["brand_name"].dropna().empty else bid,
            "mentions": int(len(grp)),
            "sov_pct": round(len(grp) / total_mentions * 100, 1),
            "avg_sentiment_score": float(round(pd.to_numeric(grp["sentiment_score"], errors="coerce").mean(), 3)),
            "positive_pct": float(round((grp["sentiment"] == "tich_cuc").mean() * 100, 1)),
            "agent_pct": float(round((grp["is_advisor"] == True).mean() * 100, 1)),
            "organic_pct": float(round((grp["is_advisor"] != True).mean() * 100, 1)),
        })
    matrix = []
    if not df_topics.empty:
        t = df_topics.copy()
        t["CommentID"] = t["CommentID"].astype(str)
        t["topic_name"] = t["topic_name"].astype(str).str.lower().str.replace("_", " ", regex=False)
        top_topics = t[t["CommentID"].isin(set(df["CommentID"]))]["topic_name"].value_counts().head(12).index.tolist()
        mt = m[["CommentID", "brand_id"]].merge(t[t["topic_name"].isin(top_topics)], on="CommentID", how="inner")
        matrix = mt.groupby(["brand_id", "topic_name"]).size().reset_index(name="count").to_dict(orient="records")
    return {"brands": sorted(rows, key=lambda r: r["mentions"], reverse=True), "topic_matrix": matrix}


def get_deep_insight_spec(start: str, end: str, groups: str = "all", brands: str = "all",
                          sentiments: str = "all", advisor_filter: str = "all",
                          base_dir: str = "./data", insurance_lines: str = "all") -> dict:
    df = _spec_joined(base_dir, start, end, groups, brands, sentiments, advisor_filter, insurance_lines)
    clusters = get_pain_clusters(start, end, "all", base_dir, advisor_filter, brands if "," not in str(brands) else "all")
    pain_df = _pain_rows(df, base_dir)
    brands_by_comment = _comment_brand_names(base_dir, set(df["CommentID"].astype(str))) if not df.empty else pd.DataFrame()
    enriched_pain = pain_df.copy()
    if not enriched_pain.empty and not brands_by_comment.empty:
        enriched_pain["CommentID"] = enriched_pain["CommentID"].astype(str)
        enriched_pain = enriched_pain.merge(brands_by_comment, on="CommentID", how="left")
    elif not enriched_pain.empty:
        enriched_pain["brand_id"] = None
        enriched_pain["brand_name"] = None

    cluster_insights = []
    brand_cluster_matrix = []
    negative_drivers = []
    representative_comments = []
    if not pain_df.empty:
        for cluster, grp in pain_df.groupby("cluster_name"):
            total = len(grp)
            neg_rate = float(round((grp["sentiment"] == "tieu_cuc").mean() * 100, 1))
            avg_score = float(round(pd.to_numeric(grp["sentiment_score"], errors="coerce").mean(), 3))
            examples = grp["pain_point"].value_counts().head(3).index.tolist()
            top_brands = []
            if not enriched_pain.empty and "brand_name" in enriched_pain:
                sub = enriched_pain[enriched_pain["cluster_name"] == cluster]
                top_brands = [
                    {"brand_name": str(k), "count": int(v)}
                    for k, v in sub["brand_name"].dropna().value_counts().head(3).items()
                ]
            severity = "high" if neg_rate >= 60 or avg_score <= -0.45 else "medium" if neg_rate >= 35 or avg_score <= -0.15 else "low"
            brand_text = ", ".join([b["brand_name"] for b in top_brands]) if top_brands else "all brands"
            example_text = ", ".join(examples[:2]) if examples else "recurring pain points"
            cluster_insights.append({
                "cluster_name": cluster,
                "title": f"{cluster}: {total} mentions",
                "summary": f"{neg_rate}% negative, avg score {avg_score}. Main signals: {example_text}. Concentrated around {brand_text}.",
                "recommended_action": "Prioritize response playbook and FAQ update." if severity == "high" else "Monitor trend and collect more customer evidence.",
                "evidence_count": int(total),
                "negative_rate": neg_rate,
                "avg_sentiment_score": avg_score,
                "severity": severity,
                "top_brands": top_brands,
                "examples": examples,
            })
            samples = grp.sort_values("sentiment_score", ascending=True).head(3)
            for _, sample in samples.iterrows():
                representative_comments.append({
                    "cluster_name": cluster,
                    "advisor_detail_labels": sample.get("advisor_detail_labels", []),
                    "CommentID": str(sample.get("CommentID", "")),
                    "Date": sample.get("Date"),
                    "Content": sample.get("Content", ""),
                    "sentiment": sample.get("sentiment", ""),
                    "sentiment_score": float(sample.get("sentiment_score")) if pd.notna(sample.get("sentiment_score")) else None,
                    "summary": sample.get("summary", ""),
                })

        cluster_insights.sort(key=lambda r: ({"high": 0, "medium": 1, "low": 2}.get(r["severity"], 3), -r["evidence_count"]))
        negative_drivers = sorted(cluster_insights, key=lambda r: (r["negative_rate"], -r["avg_sentiment_score"], r["evidence_count"]), reverse=True)[:10]

    if not enriched_pain.empty and "brand_id" in enriched_pain:
        matrix_df = enriched_pain.dropna(subset=["brand_id"])
        if not matrix_df.empty:
            top_clusters = matrix_df["cluster_name"].value_counts().head(10).index.tolist()
            top_brands = matrix_df["brand_name"].value_counts().head(10).index.tolist()
            matrix_df = matrix_df[matrix_df["cluster_name"].isin(top_clusters) & matrix_df["brand_name"].isin(top_brands)]
            brand_cluster_matrix = matrix_df.groupby(["brand_name", "cluster_name"]).size().reset_index(name="count").to_dict(orient="records")

    def phrases(slice_df: pd.DataFrame, limit: int = 25) -> list[dict]:
        import re
        stop = {"là", "và", "có", "cho", "của", "thì", "mà", "này", "được", "không", "mình", "bạn", "các"}
        counts = {}
        for text in slice_df["Content"].dropna().astype(str).head(5000):
            # Python's Unicode-aware word class handles Vietnamese safely. The
            # previous mojibake range produced an invalid character range and
            # made the entire Deep Insight endpoint return HTTP 500.
            words = [
                w.lower() for w in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
                if len(w) > 2 and w.lower() not in stop
            ]
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    counts[phrase] = counts.get(phrase, 0) + 1
        return [{"phrase": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]]

    pos_phrases = phrases(df[df["sentiment"] == "tich_cuc"]) if not df.empty else []
    neg_phrases = phrases(df[df["sentiment"] == "tieu_cuc"]) if not df.empty else []
    decision = df[df.apply(_taxonomy_intent, axis=1).isin(["muon_tu_van", "phan_van", "so_sanh_chung", "check_chi_phi"])].copy() if not df.empty else pd.DataFrame()
    decision_rows = []
    if not decision.empty:
        decision_rows = decision.sort_values("Date", ascending=False)[
            ["CommentID", "Date", "Content", "intent", "sentiment", "summary"]
        ].head(50).to_dict(orient="records")
    comment_ids = set(df["CommentID"].astype(str)) if not df.empty else set()
    return {"pain_clusters": clusters, "positive_phrases": pos_phrases,
            "negative_phrases": neg_phrases, "decision_signals": _to_records(pd.DataFrame(decision_rows)),
            "cluster_insights": _to_records(pd.DataFrame(cluster_insights)),
            "brand_cluster_matrix": _to_records(pd.DataFrame(brand_cluster_matrix)),
            "negative_drivers": _to_records(pd.DataFrame(negative_drivers)),
            "representative_comments": _to_records(pd.DataFrame(representative_comments)),
            "planning": _planning_summary(base_dir, comment_ids)}


def get_crisis_management_spec(start: str, end: str, groups: str = "all", brands: str = "all",
                               sentiments: str = "all", advisor_filter: str = "all",
                               granularity: str = "week", base_dir: str = "./data",
                               insurance_lines: str = "all") -> dict:
    df = _spec_joined(base_dir, start, end, groups, brands, sentiments, advisor_filter, insurance_lines)
    _, _, _, df_crisis, _ = _load(base_dir)
    if df.empty or df_crisis.empty:
        return {"alerts": [], "timeline": []}
    df["CommentID"] = df["CommentID"].astype(str)
    c = df_crisis.copy()
    c["CommentID"] = c["CommentID"].astype(str)
    alerts = c[c["CommentID"].isin(set(df["CommentID"]))].merge(
        df[["CommentID", "Date", "Content", "PostURL", "group_id"]], on="CommentID", how="left"
    )
    mentions_path = f"{base_dir}/brand_mentions.csv"
    if os.path.exists(mentions_path):
        bm = pd.read_csv(mentions_path).groupby("CommentID")["brand_id"].apply(lambda x: ", ".join(sorted(set(map(str, x))))).reset_index()
        bm["CommentID"] = bm["CommentID"].astype(str)
        alerts = alerts.merge(bm, on="CommentID", how="left")
    alerts["Date"] = _parse_comment_dates(alerts["Date"])
    timeline_df = _periodize(alerts.dropna(subset=["Date"]), granularity)
    timeline = timeline_df.groupby(["period", "level"]).size().reset_index(name="count").to_dict(orient="records") if not timeline_df.empty else []
    cols = ["CommentID", "brand_id", "level", "reason", "detected_at", "status", "Date", "Content", "PostURL", "group_id"]
    return {"alerts": _to_records(alerts[[c for c in cols if c in alerts.columns]].sort_values("detected_at", ascending=False)),
            "timeline": timeline}


def get_intent_topic_showcase(base_dir: str = "./data", limit_topics: int = 14,
                              brand_id: str = "all", insurance_lines: str = "all") -> dict:
    df = _joined_all(base_dir)
    raw_path = f"{base_dir}/raw_comments.csv"
    topics_path = f"{base_dir}/topics.csv"
    raw_total = len(_read_csv_cached(raw_path)) if os.path.exists(raw_path) else 0
    post_total = _count_unique_posts(base_dir)
    if not df.empty:
        if brand_id and brand_id != "all":
            mentions_path = f"{base_dir}/brand_mentions.csv"
            if os.path.exists(mentions_path):
                mentions = _read_csv_cached(mentions_path)
                ids = set(mentions[mentions["brand_id"].astype(str) == str(brand_id)]["CommentID"].astype(str))
                df = df[df["CommentID"].astype(str).isin(ids)]
            else:
                df = df.iloc[0:0]
        df = _filter_insurance_lines(df, base_dir, insurance_lines)
        post_total = _count_unique_posts(base_dir, set(df["CommentID"].astype(str))) if "CommentID" in df else 0
    if df.empty:
        return {
            "meta": {
                "base_dir": base_dir,
                "raw_total": raw_total,
                "post_total": post_total,
                "labeled_total": 0,
                "progress_pct": 0.0,
                "brand_id": brand_id,
                "insurance_lines": insurance_lines,
                "first_comment_date": None,
                "last_comment_date": None,
                "group_count": 0,
            },
            "intent_counts": [],
            "group_counts": [],
            "intent_sentiment": [],
            "negative_topics": [],
            "advisor_details": [],
            "top_topics": [],
            "intent_topic_matrix": [],
            "samples": [],
        }

    df = df.copy()
    date_values = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df else pd.Series(dtype="datetime64[ns]")
    first_date = date_values.min()
    last_date = date_values.max()
    if "PostURL" in df:
        group_keys = (
            df["PostURL"].astype(str)
            .str.split("/groups/", n=1).str[1]
            .str.split("/", n=1).str[0]
            .replace({"": pd.NA, "nan": pd.NA})
            .dropna()
        )
        group_count = int(group_keys.nunique())
    else:
        group_count = 0
    if group_count == 0 and "group_id" in df:
        group_count = int(df["group_id"].dropna().astype(str).nunique())
    df["taxonomy_intent"] = df.apply(_taxonomy_intent, axis=1)
    df["taxonomy_label"] = df["taxonomy_intent"].map(TAXONOMY_INTENT_LABELS).fillna(df["taxonomy_intent"])
    group_map = {
        "khen": "Tich cuc",
        "muon_tu_van": "Tich cuc",
        "y_kien_trung_lap": "Trung lap",
        "phan_van": "Trung lap",
        "so_sanh_chung": "Trung lap",
        "check_chi_phi": "Trung lap",
        "bao_hiem_chung": "Tieu cuc",
        "chinh_sach_quy_trinh_boi_thuong": "Tieu cuc",
        "san_pham_giai_phap": "Tieu cuc",
        "dich_vu_tu_van": "Tieu cuc",
        "spam": "Spam",
    }
    group_order = {"Tich cuc": 0, "Trung lap": 1, "Tieu cuc": 2, "Spam": 3}
    df["taxonomy_group"] = df["taxonomy_intent"].map(group_map).fillna("Khac")

    intent_counts = _intent_counts(df)
    group_counts = (
        df.groupby("taxonomy_group")
        .size()
        .reset_index(name="count")
        .assign(order=lambda x: x["taxonomy_group"].map(group_order).fillna(99))
        .sort_values("order")[["taxonomy_group", "count"]]
        .to_dict(orient="records")
    )

    sentiment_labels = {"tich_cuc": "Tich cuc", "trung_lap": "Trung lap", "tieu_cuc": "Tieu cuc"}
    sentiment_matrix = (
        df.groupby(["taxonomy_label", "sentiment"])
        .size()
        .reset_index(name="count")
    )
    sentiment_matrix["sentiment_label"] = sentiment_matrix["sentiment"].map(sentiment_labels).fillna(sentiment_matrix["sentiment"])

    negative_ids = set(NEGATIVE_TOPIC_LABELS)
    negative_df = df[df["taxonomy_intent"].isin(negative_ids)].copy()
    negative_topics = (
        negative_df.groupby(["taxonomy_intent", "taxonomy_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .to_dict(orient="records")
    )

    detail_counts = {}
    for _, row in negative_df[negative_df["taxonomy_intent"] == "dich_vu_tu_van"].iterrows():
        details = _advisor_details(row.get("Content", ""), row.get("summary", ""), row.get("pain_points", ""))
        for detail in details:
            label = ADVISOR_DETAIL_LABELS.get(detail, detail)
            detail_counts[label] = detail_counts.get(label, 0) + 1
    advisor_details = [
        {"detail": label, "count": count}
        for label, count in sorted(detail_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    if os.path.exists(topics_path):
        topics = _read_csv_cached(topics_path)
    else:
        topics = pd.DataFrame()
    if topics.empty or "CommentID" not in topics or "topic_name" not in topics:
        top_topics, matrix_rows = [], []
    else:
        topics = topics.copy()
        topics["CommentID"] = topics["CommentID"].astype(str)
        topics["topic_name"] = topics["topic_name"].astype(str).str.strip().str.lower().str.replace("_", " ", regex=False)
        topics["topic_display"] = topics["topic_name"].apply(_taxonomy_topic_name)
        labeled = df[["CommentID", "taxonomy_intent", "taxonomy_label"]].copy()
        labeled["CommentID"] = labeled["CommentID"].astype(str)
        topic_join = topics.merge(labeled, on="CommentID", how="inner")
        top_topics = (
            topic_join["topic_display"]
            .value_counts()
            .head(limit_topics)
            .reset_index()
        )
        top_topics.columns = ["topic", "count"]
        top_topics = top_topics.to_dict(orient="records")
        top_names = [row["topic"] for row in top_topics]
        matrix_rows = (
            topic_join[topic_join["topic_display"].isin(top_names)]
            .groupby(["topic_display", "taxonomy_label"])
            .size()
            .reset_index(name="count")
            .rename(columns={"topic_display": "topic", "taxonomy_label": "intent"})
            .to_dict(orient="records")
        )

    sample_cols = ["CommentID", "Date", "Content", "taxonomy_label", "taxonomy_group", "sentiment", "summary", "pain_points"]
    sample_df = df.sort_values("sentiment_score", ascending=True)
    if not negative_df.empty:
        sample_df = negative_df.sort_values("sentiment_score", ascending=True)
    samples = _to_records(sample_df[[c for c in sample_cols if c in sample_df.columns]].head(18))

    return {
        "meta": {
            "base_dir": base_dir,
            "raw_total": int(raw_total),
            "post_total": post_total,
            "labeled_total": int(len(df)),
            "progress_pct": round(len(df) / raw_total * 100, 1) if raw_total else 0.0,
            "brand_id": brand_id,
            "insurance_lines": insurance_lines,
            "first_comment_date": first_date.strftime("%Y-%m-%d") if pd.notna(first_date) else None,
            "last_comment_date": last_date.strftime("%Y-%m-%d") if pd.notna(last_date) else None,
            "group_count": group_count,
        },
        "intent_counts": intent_counts,
        "group_counts": group_counts,
        "intent_sentiment": sentiment_matrix.to_dict(orient="records"),
        "negative_topics": negative_topics,
        "advisor_details": advisor_details,
        "top_topics": top_topics,
        "intent_topic_matrix": matrix_rows,
        "samples": samples,
    }


def update_crisis_status(comment_id: str, status: str, base_dir: str = "./data") -> bool:
    path = f"{base_dir}/crisis_alerts.csv"
    if not os.path.exists(path):
        return False
    valid = {"new", "in_review", "resolved", "false_positive", "watching"}
    if status not in valid:
        raise ValueError(f"Invalid status: {status}")
    df = pd.read_csv(path)
    mask = df["CommentID"].astype(str) == str(comment_id)
    if not mask.any():
        return False
    df.loc[mask, "status"] = status
    if status == "resolved" and "resolved_at" in df.columns:
        df.loc[mask, "resolved_at"] = pd.Timestamp.now().isoformat()
    df.to_csv(path, index=False)
    return True


def _clean_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _parse_report_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), utc=True, errors="coerce")
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def _filter_report_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or "Date" not in df:
        return df
    out = df.copy()
    out["_date"] = _parse_report_dates(out["Date"])
    if start:
        out = out[out["_date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["_date"] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return out


def _report_split_filter(value: str | None) -> list[str]:
    if not value or str(value).strip() == "all":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip() and v.strip() != "all"]


def _reaction_sum(df: pd.DataFrame, cols: list[str]) -> int:
    total = 0
    for col in cols:
        if col in df:
            total += int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    return total


def get_community_report_spec(base_dir: str = "./data", brand_id: str = "prudential",
                              start: str | None = None, end: str | None = None,
                              groups: str = "all", sentiments: str = "all") -> dict:
    posts = _read_csv_cached(f"{base_dir}/raw_posts.csv")
    comments = _read_csv_cached(f"{base_dir}/raw_comments.csv")
    labels = _read_csv_cached(f"{base_dir}/ai_labels.csv")
    mentions = _read_csv_cached(f"{base_dir}/brand_mentions.csv")
    crisis = _read_csv_cached(f"{base_dir}/crisis_alerts.csv")

    group_values = _report_split_filter(groups)
    sentiment_values = _report_split_filter(sentiments)

    if not posts.empty:
        posts = posts.copy()
        posts["PostID"] = _clean_text_series(posts.get("PostID", pd.Series(dtype=str)))
        posts["group_id"] = _clean_text_series(posts.get("group_id", pd.Series(dtype=str)))
        posts = posts[posts["PostID"] != ""]
        posts = _filter_report_dates(posts, start, end)
        if group_values:
            posts = posts[posts["group_id"].isin(group_values)]
    else:
        posts = pd.DataFrame(columns=["PostID", "group_id"])

    if not comments.empty:
        comments = comments.copy()
        comments["CommentID"] = _clean_text_series(comments.get("CommentID", pd.Series(dtype=str)))
        comments["PostID"] = _clean_text_series(comments.get("PostID", pd.Series(dtype=str)))
        comments["group_id"] = _clean_text_series(comments.get("group_id", pd.Series(dtype=str)))
        comments = comments[comments["CommentID"] != ""]
        comments = _filter_report_dates(comments, start, end)
        if group_values:
            comments = comments[comments["group_id"].isin(group_values)]
    else:
        comments = pd.DataFrame(columns=["CommentID", "PostID", "group_id"])

    if not labels.empty and "CommentID" in labels and not comments.empty:
        labels = labels.copy()
        labels["CommentID"] = _clean_text_series(labels["CommentID"])
        comments = comments.merge(labels, on="CommentID", how="left", suffixes=("", "_label"))

    if not mentions.empty and {"CommentID", "brand_id"}.issubset(mentions.columns):
        mentions = mentions.copy()
        mentions["CommentID"] = _clean_text_series(mentions["CommentID"])
        mentions["brand_id"] = _clean_text_series(mentions["brand_id"]).str.lower()
    else:
        mentions = pd.DataFrame(columns=["CommentID", "brand_id"])

    if not crisis.empty and "CommentID" in crisis:
        crisis = crisis.copy()
        crisis["CommentID"] = _clean_text_series(crisis["CommentID"])
    else:
        crisis = pd.DataFrame(columns=["CommentID"])

    brand_comment_ids = set(mentions.loc[mentions["brand_id"] == str(brand_id).lower(), "CommentID"])
    brand_detection_ready = bool(brand_comment_ids)
    labeled_comments = comments[comments.get("sentiment", pd.Series(index=comments.index, dtype=object)).notna()].copy() if not comments.empty else comments
    scoped_comments = labeled_comments[labeled_comments["CommentID"].isin(brand_comment_ids)].copy() if brand_detection_ready else labeled_comments.copy()
    if sentiment_values and not scoped_comments.empty and "sentiment" in scoped_comments:
        scoped_comments = scoped_comments[_clean_text_series(scoped_comments["sentiment"]).isin(sentiment_values)]
    organic_scoped, scoped_spam_mask = _organic_comments(scoped_comments)

    sentiment_counts = {"tich_cuc": 0, "trung_lap": 0, "tieu_cuc": 0, "spam": 0}
    if not scoped_comments.empty:
        s = _clean_text_series(organic_scoped.get("sentiment", pd.Series(dtype=str)))
        sentiment_counts["tich_cuc"] = int((s == "tich_cuc").sum())
        sentiment_counts["trung_lap"] = int((s == "trung_lap").sum())
        sentiment_counts["tieu_cuc"] = int((s == "tieu_cuc").sum())
        sentiment_counts["spam"] = int(scoped_spam_mask.sum())

    total_comments = int(len(comments))
    total_posts = int(posts["PostID"].nunique()) if "PostID" in posts else 0
    labeled_count = int(len(labeled_comments))
    scoped_count = int(len(scoped_comments))
    organic_total = sentiment_counts["tich_cuc"] + sentiment_counts["trung_lap"] + sentiment_counts["tieu_cuc"]
    neg_ratio = round(sentiment_counts["tieu_cuc"] / organic_total * 100, 1) if organic_total else 0.0

    reaction_cols_good = ["Like_Count", "Love_Count", "Care_Count", "Haha_Count", "Wow_Count"]
    post_reactions = _reaction_sum(posts, ["reaction_count"])
    if post_reactions == 0:
        post_reactions = _reaction_sum(posts, reaction_cols_good + ["Sad_Count", "Angry_Count"])
    comment_reactions = _reaction_sum(comments, ["Reaction_Count"])
    if comment_reactions == 0:
        comment_reactions = _reaction_sum(comments, reaction_cols_good + ["Sad_Count", "Angry_Count"])

    groups_all = []
    if not posts.empty:
        gp = posts[[c for c in ["group_id", "group_name"] if c in posts.columns]].drop_duplicates()
        for _, row in gp.iterrows():
            gid = str(row.get("group_id", "")).strip()
            if gid:
                groups_all.append({"group_id": gid, "group_name": str(row.get("group_name", gid)).strip() or gid})
    if not comments.empty:
        for gid in sorted(set(_clean_text_series(comments["group_id"]))):
            if gid and gid not in {g["group_id"] for g in groups_all}:
                groups_all.append({"group_id": gid, "group_name": gid})
    groups_all = sorted(groups_all, key=lambda x: x["group_name"].lower())

    group_rows = []
    group_ids = sorted(set(posts.get("group_id", pd.Series(dtype=str))).union(set(comments.get("group_id", pd.Series(dtype=str)))))
    for gid in group_ids:
        if not gid:
            continue
        gp_posts = posts[posts["group_id"] == gid] if not posts.empty else posts
        gp_comments_all = comments[comments["group_id"] == gid] if not comments.empty else comments
        gp_comments = organic_scoped[organic_scoped["group_id"] == gid] if not organic_scoped.empty and "group_id" in organic_scoped else pd.DataFrame()
        ss = _clean_text_series(gp_comments.get("sentiment", pd.Series(dtype=str))) if not gp_comments.empty else pd.Series(dtype=str)
        group_rows.append({
            "group_id": gid,
            "group_name": str(gp_posts["group_name"].dropna().iloc[0]) if "group_name" in gp_posts and not gp_posts.empty and gp_posts["group_name"].dropna().shape[0] else gid,
            "posts": int(gp_posts["PostID"].nunique()) if not gp_posts.empty else 0,
            "comments": int(len(gp_comments_all)),
            "scoped_comments": int(len(gp_comments)),
            "positive": int((ss == "tich_cuc").sum()),
            "neutral": int((ss == "trung_lap").sum()),
            "negative": int((ss == "tieu_cuc").sum()),
            "post_reactions": _reaction_sum(gp_posts, ["reaction_count"]) or _reaction_sum(gp_posts, reaction_cols_good + ["Sad_Count", "Angry_Count"]),
        })

    timeline = []
    if not organic_scoped.empty and "_date" in organic_scoped:
        tmp = organic_scoped.dropna(subset=["_date"]).copy()
        if not tmp.empty:
            tmp["period"] = tmp["_date"].dt.to_period("D").astype(str)
            grouped = tmp.groupby(["period", "sentiment"]).size().reset_index(name="count")
            for period, grp in grouped.groupby("period"):
                timeline.append({
                    "period": period,
                    "positive": int(grp.loc[grp["sentiment"] == "tich_cuc", "count"].sum()),
                    "neutral": int(grp.loc[grp["sentiment"] == "trung_lap", "count"].sum()),
                    "negative": int(grp.loc[grp["sentiment"] == "tieu_cuc", "count"].sum()),
                })

    post_rows = []
    if not scoped_comments.empty and "PostID" in scoped_comments:
        post_ids = set(_clean_text_series(scoped_comments["PostID"]))
    elif not comments.empty and "PostID" in comments:
        post_ids = set(_clean_text_series(comments["PostID"]).value_counts().head(80).index)
    else:
        post_ids = set(_clean_text_series(posts.get("PostID", pd.Series(dtype=str))).head(80))
    post_ids = sorted(pid for pid in post_ids if pid)
    for pid in post_ids:
        if not pid:
            continue
        pp = posts[posts["PostID"] == pid].head(1) if not posts.empty else pd.DataFrame()
        pc_all = comments[comments["PostID"] == pid] if not comments.empty else pd.DataFrame()
        pc = organic_scoped[organic_scoped["PostID"] == pid] if not organic_scoped.empty and "PostID" in organic_scoped else pd.DataFrame()
        ss = _clean_text_series(pc.get("sentiment", pd.Series(dtype=str))) if not pc.empty else pd.Series(dtype=str)
        neg = int((ss == "tieu_cuc").sum())
        pos = int((ss == "tich_cuc").sum())
        neu = int((ss == "trung_lap").sum())
        row = pp.iloc[0] if not pp.empty else pd.Series(dtype=object)
        risky_comments = pc[pc.get("sentiment", pd.Series(index=pc.index, dtype=str)).astype(str) == "tieu_cuc"] if not pc.empty and "sentiment" in pc else pd.DataFrame()
        samples = []
        if not risky_comments.empty and "Content" in risky_comments:
            samples = [str(x).replace("\n", " ")[:160] for x in risky_comments["Content"].dropna().head(2)]
        crisis_levels = _crisis_level_counts(pc, crisis)
        angry = _reaction_sum(pp, ["Angry_Count"])
        sad = _reaction_sum(pp, ["Sad_Count"])
        negative_scores = pc.loc[ss == "tieu_cuc", "sentiment_score"] if "sentiment_score" in pc else pd.Series(dtype=float)
        score = _post_risk_score(neg, pos, neu, negative_scores, crisis_levels, angry, sad)
        post_rows.append({
            "post_id": pid,
            "group_id": str(row.get("group_id", "") or (pc_all["group_id"].iloc[0] if not pc_all.empty and "group_id" in pc_all else "")),
            "group_name": str(row.get("group_name", "") or row.get("group_id", "") or ""),
            "post_url": str(row.get("PostURL", "") or (pc_all["PostURL"].iloc[0] if not pc_all.empty and "PostURL" in pc_all else "")),
            "caption": str(row.get("PostContent", "") or "")[:280],
            "post_reactions": _reaction_sum(pp, ["reaction_count"]) or _reaction_sum(pp, reaction_cols_good + ["Sad_Count", "Angry_Count"]),
            "comment_reactions": _reaction_sum(pc_all, ["Reaction_Count"]) or _reaction_sum(pc_all, reaction_cols_good + ["Sad_Count", "Angry_Count"]),
            "comments": int(len(pc_all)),
            "scoped_comments": int(len(pc)),
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "negative_ratio": round(neg / (pos + neu + neg) * 100, 1) if (pos + neu + neg) else 0.0,
            "crisis": sum(crisis_levels.values()),
            "crisis_levels": crisis_levels,
            "risk_score": score,
            "samples": samples,
        })
    post_rows = sorted(post_rows, key=lambda r: (r["crisis"], r["negative"], r["risk_score"], r["comments"]), reverse=True)[:80]

    return {
        "meta": {
            "brand_id": brand_id,
            "brand_detection_ready": brand_detection_ready,
            "start": start,
            "end": end,
            "groups": groups,
            "sentiments": sentiments,
            "available_groups": groups_all,
            "scope_label": "Prudential mentions" if brand_detection_ready else "All labeled comments until brand detection completes",
        },
        "metrics": {
            "posts": total_posts,
            "comments": total_comments,
            "labeled_comments": labeled_count,
            "scoped_comments": scoped_count,
            "avg_comments_per_post": round(total_comments / total_posts, 1) if total_posts else 0,
            "post_reactions": post_reactions,
            "comment_reactions": comment_reactions,
            "negative_ratio": neg_ratio,
            "crisis_comments": int(scoped_comments["CommentID"].isin(set(crisis["CommentID"])).sum()) if not scoped_comments.empty and "CommentID" in scoped_comments else 0,
        },
        "sentiment_counts": sentiment_counts,
        "group_breakdown": group_rows,
        "timeline": sorted(timeline, key=lambda x: x["period"]),
        "top_posts": post_rows,
    }




def _contains_prudential(value) -> bool:
    text = _text_slug(value)
    return any(term in text for term in ["prudential", "pru", "phu hung"])


def _organic_comments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return df.copy(), pd.Series(dtype=bool)
    intent = _clean_text_series(df.get("intent", pd.Series(index=df.index, dtype=str)))
    seeded = df.get("is_seeding", pd.Series(index=df.index, dtype=object)).fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    spam_mask = (intent == "spam") | seeded
    return df.loc[~spam_mask].copy(), spam_mask


def _crisis_level_counts(df: pd.DataFrame, crisis: pd.DataFrame) -> dict[str, int]:
    counts = {"low": 0, "medium": 0, "high": 0}
    if df.empty or crisis.empty or "CommentID" not in df or "CommentID" not in crisis:
        return counts
    ids = set(_clean_text_series(df["CommentID"]))
    scoped = crisis[_clean_text_series(crisis["CommentID"]).isin(ids)]
    levels = _clean_text_series(scoped.get("level", pd.Series(dtype=str))).str.lower()
    for level in counts:
        counts[level] = int((levels == level).sum())
    return counts


def _post_risk_score(negative: int, positive: int, neutral: int, negative_scores: pd.Series,
                     crisis_levels: dict[str, int], post_angry: int, post_sad: int) -> float:
    organic_total = positive + neutral + negative
    negative_share = (negative / organic_total * 100) if organic_total else 0.0
    scores = pd.to_numeric(negative_scores, errors="coerce").dropna()
    intensity = min(100.0, float((-scores.clip(upper=0)).mean() * 100)) if not scores.empty else 0.0
    crisis_points = min(100.0, crisis_levels.get("low", 0) * 5 + crisis_levels.get("medium", 0) * 25 + crisis_levels.get("high", 0) * 100)
    reaction_points = min(100.0, post_angry * 5 + post_sad * 2)
    return round(min(100.0, negative_share * 0.45 + intensity * 0.25 + crisis_points * 0.20 + reaction_points * 0.10), 1)


def _recommend_seeding(negative_ratio: float, has_negative_pru_mention: bool) -> str:
    if negative_ratio >= 1.5 and has_negative_pru_mention:
        return "Urgent"
    if negative_ratio >= 1.0:
        return "Normal"
    return "Following"


def get_community_table_report(base_dir: str = "./data", brand_id: str = "prudential",
                               start: str | None = None, end: str | None = None,
                               groups: str = "all", sentiments: str = "all",
                               limit: int = 500) -> dict:
    posts = _read_csv_cached(f"{base_dir}/raw_posts.csv")
    comments = _read_csv_cached(f"{base_dir}/raw_comments.csv")
    labels = _read_csv_cached(f"{base_dir}/ai_labels.csv")
    crisis = _read_csv_cached(f"{base_dir}/crisis_alerts.csv")
    mentions = _read_csv_cached(f"{base_dir}/brand_mentions.csv")

    group_values = _report_split_filter(groups)
    sentiment_values = _report_split_filter(sentiments)
    limit = max(1, min(int(limit or 500), 5000))

    if not posts.empty:
        posts = posts.copy()
        posts["PostID"] = _clean_text_series(posts.get("PostID", pd.Series(dtype=str)))
        posts["group_id"] = _clean_text_series(posts.get("group_id", pd.Series(dtype=str)))
        posts = posts[posts["PostID"] != ""]
        posts = _filter_report_dates(posts, start, end)
        if group_values:
            posts = posts[posts["group_id"].isin(group_values)]
    else:
        posts = pd.DataFrame(columns=["PostID", "group_id"])

    if not comments.empty:
        comments = comments.copy()
        comments["CommentID"] = _clean_text_series(comments.get("CommentID", pd.Series(dtype=str)))
        comments["PostID"] = _clean_text_series(comments.get("PostID", pd.Series(dtype=str)))
        comments["group_id"] = _clean_text_series(comments.get("group_id", pd.Series(dtype=str)))
        comments = comments[comments["CommentID"] != ""]
        comments = _filter_report_dates(comments, start, end)
        if group_values:
            comments = comments[comments["group_id"].isin(group_values)]
    else:
        comments = pd.DataFrame(columns=["CommentID", "PostID", "group_id"])

    if not labels.empty and "CommentID" in labels and not comments.empty:
        labels = labels.copy()
        labels["CommentID"] = _clean_text_series(labels["CommentID"])
        comments = comments.merge(labels, on="CommentID", how="left", suffixes=("", "_label"))

    labeled = comments[comments.get("sentiment", pd.Series(index=comments.index, dtype=object)).notna()].copy() if not comments.empty else comments
    if sentiment_values and not labeled.empty and "sentiment" in labeled:
        labeled = labeled[_clean_text_series(labeled["sentiment"]).isin(sentiment_values)]

    if not crisis.empty and "CommentID" in crisis:
        crisis_ids = set(_clean_text_series(crisis["CommentID"]))
    else:
        crisis_ids = set()

    if not mentions.empty and {"CommentID", "brand_id"}.issubset(mentions.columns):
        prudential_comment_ids = set(
            _clean_text_series(mentions.loc[
                _clean_text_series(mentions["brand_id"]).str.lower() == "prudential",
                "CommentID",
            ])
        )
    else:
        prudential_comment_ids = set()

    post_lookup = {}
    if not posts.empty:
        for _, row in posts.drop_duplicates(subset=["PostID"]).iterrows():
            post_lookup[str(row.get("PostID", ""))] = row

    group_lookup = {}
    if not posts.empty and "group_name" in posts:
        gp = posts[[c for c in ["group_id", "group_name"] if c in posts.columns]].drop_duplicates()
        for _, row in gp.iterrows():
            gid = str(row.get("group_id", "")).strip()
            if gid:
                group_lookup[gid] = str(row.get("group_name", gid)).strip() or gid

    available_groups = [
        {"group_id": gid, "group_name": name}
        for gid, name in sorted(group_lookup.items(), key=lambda item: item[1].lower())
    ]

    raw_comment_counts = comments.groupby("PostID").size().to_dict() if not comments.empty and "PostID" in comments else {}
    comment_reactions = {}
    if not comments.empty and "PostID" in comments:
        reaction_cols = ["Reaction_Count", "Like_Count", "Love_Count", "Care_Count", "Haha_Count", "Wow_Count", "Sad_Count", "Angry_Count"]
        available = [c for c in reaction_cols if c in comments.columns]
        if available:
            tmp = comments[["PostID"] + available].copy()
            for col in available:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
            comment_reactions = tmp.groupby("PostID")[available].sum().sum(axis=1).to_dict()

    rows = []
    post_ids = set(posts.get("PostID", pd.Series(dtype=str))).union(set(comments.get("PostID", pd.Series(dtype=str))))
    if not labeled.empty and "PostID" in labeled:
        post_ids = post_ids.union(set(labeled["PostID"]))

    for idx, pid in enumerate(sorted(pid for pid in post_ids if str(pid).strip()), start=1):
        pid = str(pid).strip()
        post = post_lookup.get(pid, pd.Series(dtype=object))
        post_comments = comments[comments["PostID"] == pid] if not comments.empty and "PostID" in comments else pd.DataFrame()
        post_labeled = labeled[labeled["PostID"] == pid] if not labeled.empty and "PostID" in labeled else pd.DataFrame()
        organic_labeled, spam_mask = _organic_comments(post_labeled)
        sentiments = _clean_text_series(organic_labeled.get("sentiment", pd.Series(dtype=str))) if not organic_labeled.empty else pd.Series(dtype=str)
        intents = _clean_text_series(post_labeled.get("intent", pd.Series(dtype=str))) if not post_labeled.empty else pd.Series(dtype=str)

        positive = int((sentiments == "tich_cuc").sum())
        neutral = int((sentiments == "trung_lap").sum())
        negative = int((sentiments == "tieu_cuc").sum())
        spam = int(spam_mask.sum())

        base = positive + neutral
        negative_ratio = round(negative / base, 2) if base else (999.0 if negative else 0.0)
        all_crisis_levels = _crisis_level_counts(organic_labeled, crisis)
        post_mentions_prudential = _contains_prudential(post.get("PostContent", ""))
        if not organic_labeled.empty:
            comment_mentions_prudential = (
                _clean_text_series(organic_labeled["CommentID"]).isin(prudential_comment_ids)
                | organic_labeled.get("Content", pd.Series("", index=organic_labeled.index)).apply(_contains_prudential)
            )
            prudential_crisis_scope = organic_labeled[
                comment_mentions_prudential | bool(post_mentions_prudential)
            ]
        else:
            prudential_crisis_scope = organic_labeled
        crisis_levels = _crisis_level_counts(prudential_crisis_scope, crisis)
        crisis_count = sum(crisis_levels.values())

        negative_mentions = []
        negative_prudential_count = 0
        prudential_mentions = []
        if not post_labeled.empty and "Content" in post_labeled:
            pru_scope = post_labeled[post_labeled["Content"].apply(_contains_prudential).astype(bool)]
            prudential_mentions = [str(x).replace("\n", " ")[:220] for x in pru_scope["Content"].dropna()]
            neg_scope = organic_labeled[sentiments == "tieu_cuc"]
            if brand_id and str(brand_id).lower() == "prudential" and not neg_scope.empty:
                brand_mask = (
                    _clean_text_series(neg_scope["CommentID"]).isin(prudential_comment_ids)
                    | neg_scope["Content"].apply(_contains_prudential).astype(bool)
                )
                neg_scope = neg_scope[brand_mask]
            negative_prudential_count = len(neg_scope)
            if not neg_scope.empty and "Content" in neg_scope:
                negative_mentions = [str(x).replace("\n", " ")[:220] for x in neg_scope["Content"].dropna().head(3)]

        subtypes = []
        if not intents.empty:
            counts = intents.replace({"": pd.NA, "nan": pd.NA}).dropna().value_counts().head(4)
            subtypes = [f"{TAXONOMY_INTENT_LABELS.get(k, k)}: {int(v)}" for k, v in counts.items()]

        post_df = pd.DataFrame([post])
        positive_reaction_group = _reaction_sum(post_df, ["Like_Count", "Love_Count", "Care_Count", "Haha_Count", "Wow_Count"])
        sad = _reaction_sum(post_df, ["Sad_Count"])
        angry = _reaction_sum(post_df, ["Angry_Count"])
        comment_count = int(raw_comment_counts.get(pid, len(post_comments)))
        negative_scores = organic_labeled.loc[sentiments == "tieu_cuc", "sentiment_score"] if "sentiment_score" in organic_labeled else pd.Series(dtype=float)
        risk_score = _post_risk_score(negative, positive, neutral, negative_scores, all_crisis_levels, angry, sad)
        comment_details = []
        for _, comment_row in post_labeled.iterrows():
            intent_value = str(comment_row.get("intent", "") or "").strip()
            seeded_value = str(comment_row.get("is_seeding", "") or "").strip().lower() in {"true", "1", "yes"}
            is_spam = intent_value == "spam" or seeded_value
            sentiment_value = str(comment_row.get("sentiment", "") or "").strip()
            content_value = str(comment_row.get("Content", "") or "").strip()
            mentions_pru = _contains_prudential(content_value)
            comment_details.append({
                "comment_id": str(comment_row.get("CommentID", "") or ""),
                "content": content_value,
                "author": str(comment_row.get("Author", "") or ""),
                "date": str(comment_row.get("Date", "") or ""),
                "permalink": str(comment_row.get("Permalink", "") or ""),
                "sub_loai": TAXONOMY_INTENT_LABELS.get(intent_value, intent_value),
                "positive": bool(not is_spam and sentiment_value == "tich_cuc"),
                "neutral": bool(not is_spam and sentiment_value == "trung_lap"),
                "negative": bool(not is_spam and sentiment_value == "tieu_cuc"),
                "spam": bool(is_spam),
                "mentions_prudential": bool(mentions_pru),
                "negative_prudential_mention": bool(mentions_pru and not is_spam and sentiment_value == "tieu_cuc"),
            })
        group_id = str(post.get("group_id", "") or (post_comments["group_id"].iloc[0] if not post_comments.empty and "group_id" in post_comments else ""))

        rows.append({
            "stt": idx,
            "post_id": pid,
            "group_id": group_id,
            "group_name": str(post.get("group_name", "") or group_lookup.get(group_id, group_id)),
            "link_post": str(post.get("PostURL", "") or (post_comments["PostURL"].iloc[0] if not post_comments.empty and "PostURL" in post_comments else "")),
            "caption": str(post.get("PostContent", "") or "")[:600],
            "reaction_positive_group": positive_reaction_group,
            "reaction_sad": sad,
            "reaction_angry": angry,
            "comment": comment_count,
            "comment_reaction": int(comment_reactions.get(pid, 0)),
            "sub_loai": "; ".join(subtypes),
            "sentiment_positive": positive,
            "sentiment_neutral": neutral,
            "sentiment_negative": negative,
            "sentiment_spam": spam,
            "sentiment_total": positive + neutral + negative + spam,
            "prudential_mentions": prudential_mentions,
            "prudential_mention_count": len(prudential_mentions),
            "negative_prudential_mentions": negative_mentions,
            "negative_prudential_count": negative_prudential_count,
            "negative_ratio": negative_ratio,
            "negative_ratio_formula": f"{negative} / ({positive} + {neutral})",
            "seeding_recommendation": _recommend_seeding(negative_ratio, bool(negative_mentions)),
            "pillar": "",
            "crisis_comments": crisis_count,
            "crisis_levels": crisis_levels,
            "risk_score": risk_score,
            "comment_details": comment_details,
        })

    rows = sorted(rows, key=lambda r: (r["crisis_comments"], r["sentiment_negative"], r["risk_score"], r["comment"]), reverse=True)[:limit]
    for idx, row in enumerate(rows, start=1):
        row["stt"] = idx

    return {
        "meta": {
            "brand_id": brand_id,
            "start": start,
            "end": end,
            "groups": groups,
            "sentiments": sentiments,
            "available_groups": available_groups,
            "row_count": len(rows),
            "crisis_scope": "prudential_mentions_only",
            "negative_scope": "all_brands",
        },
        "rows": rows,
    }
