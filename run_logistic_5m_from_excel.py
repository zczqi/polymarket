import os
import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


EXCEL_PATH = "clean_orderbook_dataset revise.xlsx"

HORIZON_MINUTES = 5
TOLERANCE_SECONDS = 180

TARGET_COL = "y_up_5m"

FEATURE_COLS = [
    "abs_delta",
    "relative_spread",
    "imbalance_top5",
    "depth_bid_top5",
    "depth_ask_top5",
]

REQUIRED_COLS = [
    "token_id",
    "ts_utc",
    "mid_price",
    "microprice",
    "relative_spread",
    "imbalance_top5",
    "depth_bid_top5",
    "depth_ask_top5",
]


import re
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

def parse_time_series(s: pd.Series) -> pd.Series:
    """
    Ultra-robust UTC timestamp parser.
    Handles:
    - real datetimes
    - Excel serial numbers
    - messy strings like 2026-02-04T07:44:04.402682+00:00
    - hidden characters / extra suffixes
    """
    if is_datetime64_any_dtype(s):
        return pd.to_datetime(s, utc=True, errors="coerce")

    if is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1899-12-30", utc=True, errors="coerce")

    # Normalize to string
    raw = s.astype("string")

    # Remove hidden characters and trim
    raw = raw.str.replace("\u200b", "", regex=False)   # zero-width space
    raw = raw.str.replace("\ufeff", "", regex=False)   # BOM
    raw = raw.str.strip()

    # First attempt: direct parse
    ts1 = pd.to_datetime(raw, utc=True, errors="coerce")

    # If mostly failed, do regex extraction then parse
    if float(ts1.isna().mean()) > 0.5:
        pat = r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        extracted = raw.str.extract(pat)
        iso = extracted[0].fillna("") + " " + extracted[1].fillna("")
        iso = iso.str.strip()
        ts2 = pd.to_datetime(iso, utc=True, errors="coerce")

        # Some sheets store Excel serials as strings
        if float(ts2.isna().mean()) > 0.5:
            nums = pd.to_numeric(raw, errors="coerce")
            ts3 = pd.to_datetime(nums, unit="D", origin="1899-12-30", utc=True, errors="coerce")
            # pick best
            best = ts1
            if ts2.notna().sum() > best.notna().sum():
                best = ts2
            if ts3.notna().sum() > best.notna().sum():
                best = ts3
            return best

        # pick best between ts1 and ts2
        return ts2 if ts2.notna().sum() > ts1.notna().sum() else ts1

    return ts1



def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def attach_future_mid(df: pd.DataFrame, horizon_min: int, tol_s: int) -> pd.DataFrame:
    horizon_ns = int(horizon_min * 60 * 1e9)
    tol_ns = int(tol_s * 1e9)

    out = []
    for token_id, g in df.groupby("token_id", sort=False):
        g = g.sort_values("ts").reset_index(drop=True)

        ts_ns = g["ts"].astype("int64").to_numpy()
        mid = g["mid_price"].to_numpy()

        target_ns = ts_ns + horizon_ns
        j = np.searchsorted(ts_ns, target_ns, side="left")

        mid_fut = np.full(len(g), np.nan, dtype=float)

        valid = j < len(g)
        jv = j[valid]

        diff = np.abs(ts_ns[jv] - target_ns[valid])
        ok = diff <= tol_ns

        left_idx = np.where(valid)[0][ok]
        chosen = jv[ok]

        mid_fut[left_idx] = mid[chosen]
        g["mid_fut"] = mid_fut

        out.append(g)

    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0].copy()


def build_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    df = df_raw[REQUIRED_COLS].copy()

    # token_id as string to prevent scientific notation grouping issues
    df["token_id"] = df["token_id"].astype(str).str.strip()

    # parse time
    df["ts"] = parse_time_series(df["ts_utc"]) 
    
    print("ts parse bad ratio:", float(df["ts"].isna().mean()))
    

    # numeric
    df = coerce_numeric(df, ["mid_price", "microprice", "relative_spread",
                             "imbalance_top5", "depth_bid_top5", "depth_ask_top5"])

    # feature engineered
    df["abs_delta"] = (df["microprice"] - df["mid_price"]).abs()

    # drop invalid
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["token_id", "ts", "mid_price", "microprice"]).copy()

    # future mid and label
    df = attach_future_mid(df, HORIZON_MINUTES, TOLERANCE_SECONDS)
    df = df.dropna(subset=["mid_fut"]).copy()
    df[TARGET_COL] = (df["mid_fut"] > df["mid_price"]).astype(int)

    # keep complete rows
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    df = df.sort_values(["token_id", "ts"]).reset_index(drop=True)
    return df


def train_logistic(ds: pd.DataFrame):
    if len(ds) == 0:
        print("No rows after labeling.")
        return

    X = ds[FEATURE_COLS].copy()
    y = ds[TARGET_COL].astype(int).copy()

    if y.nunique() < 2:
        print("Only one class in y. Cannot train.")
        print(y.value_counts())
        return

    cut = int(len(ds) * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, p) if y_test.nunique() == 2 else np.nan

    print("")
    print("Rows used:", len(ds))
    print("Train:", len(X_train), "Test:", len(X_test))
    print("Test positive rate:", float(y_test.mean()))
    print("ACC:", float(acc))
    print("AUC:", float(auc))
    print("")
    print("Confusion matrix")
    print(confusion_matrix(y_test, pred))
    print("")
    print(classification_report(y_test, pred, digits=4))

    coefs = model.named_steps["clf"].coef_[0]
    coef_tbl = pd.DataFrame({"feature": FEATURE_COLS, "coef": coefs}).sort_values("coef", ascending=False)
    print("\nCoefficients")
    print(coef_tbl.to_string(index=False))


def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Cannot find {EXCEL_PATH} in current folder")

    # only read required columns (fast)
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl", usecols=REQUIRED_COLS)

    ds = build_dataset(df)

    print("dataset built:", len(ds), "rows")
    print("tokens:", ds["token_id"].nunique() if len(ds) else 0)

    train_logistic(ds)


if __name__ == "__main__":
    main()
