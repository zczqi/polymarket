import argparse
import glob
import os
import sys
from decimal import Decimal, InvalidOperation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 5-minute logistic-regression baseline from an Excel orderbook export."
    )
    parser.add_argument(
        "--excel",
        default=None,
        help=(
            "Path to source .xlsx file. If omitted, the script tries a few defaults and then "
            "falls back to the first .xlsx file in the current folder."
        ),
    )
    parser.add_argument("--horizon-minutes", type=int, default=5)
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=None,
        help="Optional horizon in seconds (overrides --horizon-minutes when provided).",
    )
    parser.add_argument("--tolerance-seconds", type=int, default=180)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print row counts at each dataset-building step for debugging label issues.",
    )
    return parser.parse_args()


def resolve_excel_path(explicit_path: str | None) -> str:
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Cannot find Excel file: {explicit_path}")

    default_candidates = [
        "clean_orderbook_dataset revise.xlsx",
        "clean_database_revise.xlsx",
        "clean database revise.xlsx",
    ]
    for candidate in default_candidates:
        if os.path.exists(candidate):
            return candidate

    xlsx_files = sorted(glob.glob("*.xlsx"))
    if len(xlsx_files) == 1:
        return xlsx_files[0]

    if len(xlsx_files) > 1:
        raise FileNotFoundError(
            "Multiple .xlsx files found. Re-run with --excel <file>. Candidates: "
            + ", ".join(xlsx_files)
        )

    raise FileNotFoundError("No .xlsx files found in the current folder.")


def main():
    args = parse_args()

    try:
        import numpy as np
        import pandas as pd
        from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required dependency"
        print(
            f"Missing dependency: {missing}.\n"
            "Install requirements first, for example:\n"
            "  python -m pip install numpy pandas scikit-learn openpyxl",
            file=sys.stderr,
        )
        raise SystemExit(1)

    EXCEL_PATH = resolve_excel_path(args.excel)

    horizon_seconds = int(args.horizon_seconds) if args.horizon_seconds is not None else int(args.horizon_minutes * 60)
    tolerance_seconds = int(args.tolerance_seconds)

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

    def parse_time_series(s: pd.Series) -> pd.Series:
        if is_datetime64_any_dtype(s):
            return pd.to_datetime(s, utc=True, errors="coerce")

        if is_numeric_dtype(s):
            return pd.to_datetime(s, unit="D", origin="1899-12-30", utc=True, errors="coerce")

        raw = s.astype("string")
        raw = raw.str.replace("\u200b", "", regex=False)
        raw = raw.str.replace("\ufeff", "", regex=False)
        raw = raw.str.strip()

        ts1 = pd.to_datetime(raw, utc=True, errors="coerce")

        if float(ts1.isna().mean()) > 0.5:
            pat = r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
            extracted = raw.str.extract(pat)
            iso = extracted[0].fillna("") + " " + extracted[1].fillna("")
            iso = iso.str.strip()
            ts2 = pd.to_datetime(iso, utc=True, errors="coerce")

            if float(ts2.isna().mean()) > 0.5:
                nums = pd.to_numeric(raw, errors="coerce")
                ts3 = pd.to_datetime(nums, unit="D", origin="1899-12-30", utc=True, errors="coerce")
                best = ts1
                if ts2.notna().sum() > best.notna().sum():
                    best = ts2
                if ts3.notna().sum() > best.notna().sum():
                    best = ts3
                return best

            return ts2 if ts2.notna().sum() > ts1.notna().sum() else ts1

        return ts1

    def coerce_numeric(df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df


    def normalize_token_id(s: pd.Series) -> pd.Series:
        # Excel often stores huge on-chain IDs in scientific notation (e.g., 2.4286E+76).
        # Tiny float formatting differences can split one token into many fake groups.
        def _one(v):
            if v is None:
                return pd.NA
            txt = str(v).strip()
            if txt == "" or txt.lower() in {"nan", "none", "<na>"}:
                return pd.NA
            try:
                d = Decimal(txt)
                if d == d.to_integral_value():
                    return format(d.quantize(Decimal(1)), "f")
                return format(d.normalize(), "f")
            except (InvalidOperation, ValueError):
                return txt

        return s.map(_one).astype("string")

    def attach_future_mid(df: pd.DataFrame, horizon_s: int, tol_s: int) -> pd.DataFrame:
        horizon = pd.to_timedelta(horizon_s, unit="s")
        tolerance = pd.to_timedelta(tol_s, unit="s")

        # Run merge_asof per token to avoid global key-sorting constraints across pandas versions.
        out_parts = []
        for token_id, g in df.groupby("token_id", sort=False):
            g = g.sort_values("ts").reset_index(drop=True).copy()
            g["ts_target"] = g["ts"] + horizon

            right = g[["ts", "mid_price"]].rename(columns={"ts": "ts_match", "mid_price": "mid_fut"})
            right = right.sort_values("ts_match").reset_index(drop=True)

            left = g.sort_values("ts_target").reset_index().rename(columns={"index": "_orig_idx"})
            merged = pd.merge_asof(
                left,
                right,
                left_on="ts_target",
                right_on="ts_match",
                direction="nearest",
                tolerance=tolerance,
            )

            merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx", "ts_target", "ts_match"])
            out_parts.append(merged)

        out = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
        return out

    def build_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Missing required columns in Excel: {missing}")

        df = df_raw[REQUIRED_COLS].copy()
        if args.verbose:
            print("rows raw:", len(df))

        df["token_id"] = normalize_token_id(df["token_id"])
        df["ts"] = parse_time_series(df["ts_utc"])

        print("ts parse bad ratio:", float(df["ts"].isna().mean()))
        if args.verbose:
            print("rows with valid ts:", int(df["ts"].notna().sum()))
            print("horizon_seconds:", horizon_seconds)
            print("tolerance_seconds:", tolerance_seconds)

        df = coerce_numeric(
            df,
            [
                "mid_price",
                "microprice",
                "relative_spread",
                "imbalance_top5",
                "depth_bid_top5",
                "depth_ask_top5",
            ],
        )

        df["abs_delta"] = (df["microprice"] - df["mid_price"]).abs()

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["token_id", "ts", "mid_price", "microprice"]).copy()
        if args.verbose:
            print("rows after required non-null filters:", len(df))
            tok_counts = df["token_id"].value_counts(dropna=False)
            print("unique token_id after normalization:", int(df["token_id"].nunique(dropna=True)))
            print("top token counts:")
            print(tok_counts.head(10).to_string())
            # quick cadence diagnostics to debug future-match failures
            tmp = df.sort_values(["token_id", "ts"]).copy()
            tmp["gap_s"] = tmp.groupby("token_id")["ts"].diff().dt.total_seconds()
            gap_stats = tmp.groupby("token_id")["gap_s"].median().head(5)
            if len(gap_stats):
                print("median time gap(s) by token (top 5):")
                print(gap_stats.to_string())

        df = attach_future_mid(df, horizon_seconds, tolerance_seconds)
        if args.verbose:
            print("rows after future match attempt:", len(df))
            print("rows with matched future mid:", int(df["mid_fut"].notna().sum()))
            if len(df):
                match_rate = float(df["mid_fut"].notna().mean())
                print("future match rate:", round(match_rate, 6))
        df = df.dropna(subset=["mid_fut"]).copy()
        df[TARGET_COL] = (df["mid_fut"] > df["mid_price"]).astype(int)

        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
        if args.verbose:
            print("rows after feature/target non-null filters:", len(df))
            print("target distribution:")
            print(df[TARGET_COL].value_counts(dropna=False).to_string())

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

    print(f"Using Excel file: {EXCEL_PATH}")
    df = pd.read_excel(
        EXCEL_PATH,
        engine="openpyxl",
        usecols=REQUIRED_COLS,
        converters={
            "token_id": lambda x: "" if pd.isna(x) else str(x),
            "ts_utc": lambda x: "" if pd.isna(x) else str(x),
        },
    )

    ds = build_dataset(df)

    print("dataset built:", len(ds), "rows")
    print("tokens:", ds["token_id"].nunique() if len(ds) else 0)

    train_logistic(ds)


if __name__ == "__main__":
    main()
