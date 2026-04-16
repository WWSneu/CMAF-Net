import re

import pandas as pd


def split_into_clauses(text):
    if pd.isna(text):
        return [""]

    clauses = re.split(r"[，。！？；;,.!?]", str(text))
    clauses = [c.strip() for c in clauses if c.strip()]

    if len(clauses) == 0:
        clauses = [str(text).strip()]

    return clauses[:16]


def detect_city_column(df):
    city_candidates = ["city", "shop_city", "region", "district_city"]
    for c in city_candidates:
        if c in df.columns:
            return c
    return None


def ensure_column(df, col, default_value=0):
    if col not in df.columns:
        df[col] = default_value
    return df


def build_feature_dataframe(csv_path="data/merged_data.csv"):
    df = pd.read_csv(csv_path).copy()

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("merged_data.csv 必须至少包含 text 和 label 列。")

    df["label"] = df["label"].astype(int)

    city_col = detect_city_column(df)
    if city_col is None:
        df["city"] = "GLOBAL"
        city_col = "city"

    base_numeric_cols = [
        "vip", "discount", "overall", "taste", "environment", "service", "ingredient",
        "consumption", "food", "picture", "like", "response", "interaction"
    ]
    for col in base_numeric_cols:
        df = ensure_column(df, col, 0)

    df["time"] = pd.to_datetime(df.get("time", pd.Series([None] * len(df))), errors="coerce")
    df["publish_hour"] = df["time"].dt.hour.fillna(12).astype(int)
    df["is_weekend"] = df["time"].dt.dayofweek.fillna(0).astype(int).apply(lambda x: 1 if x >= 5 else 0)

    df["text"] = df["text"].astype(str)
    df["text_length"] = df["text"].apply(len)
    df["exclamation_count"] = df["text"].apply(lambda x: x.count("!") + x.count("！") + x.count("?") + x.count("？"))
    df["has_picture"] = df["picture"].apply(lambda x: 1 if x > 0 else 0)
    df["is_extreme_rating"] = df["overall"].apply(lambda x: 1 if x in [1, 5] else 0)

    df["rating_mean"] = df[["taste", "environment", "service", "ingredient"]].mean(axis=1)
    df["rating_var"] = df[["overall", "taste", "environment", "service", "ingredient"]].var(axis=1)
    df["rating_dev"] = df["overall"] - df[["taste", "environment", "service", "ingredient"]].mean(axis=1)
    df["has_consumption"] = df["consumption"].apply(lambda x: 1 if x > 0 else 0)
    df["valid_post_hour"] = df["publish_hour"].apply(lambda h: 1 if 6 <= h <= 23 else 0)

    df["clauses"] = df["text"].apply(split_into_clauses)

    meta_cols = [
        "vip", "discount",
        "overall", "taste", "environment", "service", "ingredient",
        "consumption", "has_consumption",
        "food", "picture", "has_picture",
        "like", "response", "interaction",
        "text_length", "exclamation_count",
        "is_extreme_rating", "publish_hour", "is_weekend", "valid_post_hour",
        "rating_mean", "rating_var", "rating_dev"
    ]

    for col in meta_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print("数据量:", len(df))
    print("标签分布:")
    print(df["label"].value_counts())
    print("使用的 city 列:", city_col)
    print("metadata 列数:", len(meta_cols))

    return df, meta_cols, city_col
