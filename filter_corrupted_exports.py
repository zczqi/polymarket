"""
===============================================================================
filter_corrupted_exports.py
===============================================================================

PURPOSE
-------
This script serves two major purposes:

1) Export merged orderbook data for a specific market_id
2) Export per-market metadata (row counts) to identify the largest markets
   for model training

Your database tables:
    Snapshots table : orderbook_snapshots
    Features table  : features_orderbook

===============================================================================
OPERATING MODES
===============================================================================

MODE A — MARKET EXPORT MODE (default)
--------------------------------------
Exports merged snapshot + feature data for a single market.

By default:
    - Corrupted rows are filtered out
    - Clean dataset is exported
    - Flagged rows are exported separately

Optional:
    --no-filter  → disables filtering and exports ALL rows
                   (recommended for debugging)

Requires:
    --market-id


MODE B — MARKET METADATA MODE
------------------------------
Exports metadata for ALL markets in the database.

This mode:
    - Counts rows per market_id
    - Counts distinct token_id per market
    - Computes timestamp coverage window
    - Sorts markets by size (largest first)
    - Exports to CSV

Does NOT require:
    --market-id

Activated with:
    --export-market-metadata


===============================================================================
USAGE EXAMPLES
===============================================================================

# 1) Export a specific market (filter ON, default behavior)
python filter_corrupted_exports.py --db polymarket.db --market-id 654415


# 2) Export a specific market (NO filtering, debugging mode)
python filter_corrupted_exports.py --db polymarket.db --market-id 654415 --no-filter --out-clean raw_debug_export.csv


# 3) Export a market for a specific time range
python filter_corrupted_exports.py \
    --market-id 654415 \
    --start "2026-02-01T00:00:00Z" \
    --end   "2026-02-10T00:00:00Z"


# 4) Export metadata for ALL markets (to find largest dataset)
python filter_corrupted_exports.py \
    --db polymarket.db \
    --export-market-metadata \
    --market-metadata-out market_metadata.csv


# 5) Export metadata using non-default table names
python filter_corrupted_exports.py \
    --db polymarket.db \
    --snapshots-table orderbook_snapshots \
    --features-table features_orderbook \
    --export-market-metadata


===============================================================================
KEY OPTIONS
===============================================================================

--market-id
    Market ID to export.
    Required unless --export-market-metadata is used.

--export-market-metadata
    Export per-market dataset size statistics and exit.

--market-metadata-out
    Output file for metadata CSV (default: market_metadata.csv)

--no-filter
    Disable corrupted-data filtering entirely.
    Export everything.
    Recommended when debugging alignment issues.

--interval
    Expected snapshot interval in seconds.
    If omitted, inferred automatically per token.

--tolerance
    Relative tolerance for interval validation (default ±35%).

--grid-slack
    Absolute seconds allowed off expected interval grid.

--max-levels
    Number of bid/ask levels expanded from JSON (default 10).

--snapshots-table
    Snapshot table name (default: orderbook_snapshots)

--features-table
    Feature table name (default: features_orderbook)


===============================================================================
METADATA OUTPUT STRUCTURE
===============================================================================

The metadata CSV includes:

    market_id
    snapshot_rows        (total rows in orderbook_snapshots)
    feature_rows         (total rows in features_orderbook)
    snapshot_tokens      (distinct token_id count)
    snapshot_min_ts      (earliest timestamp)
    snapshot_max_ts      (latest timestamp)

Sorted by:
    snapshot_rows (descending)

Use this file to select the market with the largest dataset
for model training.


===============================================================================
DEBUGGING STRATEGY
===============================================================================

If:
    - Row counts look incorrect
    - Time coverage seems off
    - Merge alignment seems wrong

Step 1:
    Run metadata export to see total data per market.

Step 2:
    Run market export with --no-filter.

Step 3:
    Compare raw export vs filtered export.

This isolates whether the issue is:
    (A) ingestion issue
    (B) merge key mismatch
    (C) filtering too strict
    (D) timestamp irregularity

===============================================================================
"""


from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    db: str
    market_id: str
    start_utc: Optional[str]
    end_utc: Optional[str]
    out_clean: str
    out_flagged: str
    interval: Optional[float]        # expected interval seconds (if None, infer per token)
    tolerance: float                 # relative tolerance around interval (e.g., 0.35 = +/-35%)
    min_samples: int                 # minimum rows per token to attempt interval checks
    grid_slack_s: Optional[float]    # absolute slack allowed off-grid, if set
    enable_filter: bool              # if False: export-only mode (debug)
    snapshots_table: str
    features_table: str


# -----------------------------
# Time parsing helpers
# -----------------------------
def _parse_iso_or_naive(s: str) -> datetime:
    """
    Parse either:
    - ISO8601 with Z / offset, e.g. 2026-02-01T00:00:00Z or 2026-02-01T00:00:00+00:00
    - naive "2026-02-01 00:00:00" or "2026-02-01T00:00:00"
    """
    s = s.strip()
    if s.endswith("Z"):
        # Python doesn't parse Z directly in fromisoformat
        s2 = s[:-1] + "+00:00"
        return datetime.fromisoformat(s2)
    return datetime.fromisoformat(s.replace(" ", "T"))


def parse_time_to_utc_iso(dt_str: Optional[str], tz_name: str) -> Optional[str]:
    """
    Convert user-provided time string to an ISO UTC string (with 'Z').

    - If dt_str is None -> None
    - If dt_str has timezone info -> convert to UTC
    - If dt_str is naive -> interpret in tz_name (default UTC), then convert to UTC
    """
    if not dt_str:
        return None

    dt = _parse_iso_or_naive(dt_str)

    if dt.tzinfo is None:
        if tz_name.upper() == "UTC":
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            if ZoneInfo is None:
                raise RuntimeError("zoneinfo not available; use timezone-aware times or set --tz UTC")
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))

    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------
# DB helpers
# -----------------------------
def assert_table_exists(conn: sqlite3.Connection, tname: str) -> None:
    cur = conn.cursor()
    ok = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (tname,),
    ).fetchone()
    if ok:
        return
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    available = ", ".join(r[0] for r in rows)
    raise RuntimeError(f"Table '{tname}' not found. Available tables: {available}")


def _read_sql_df(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...]) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def read_tables(
    conn: sqlite3.Connection,
    market_id: str,
    start_utc: Optional[str],
    end_utc: Optional[str],
    snapshots_table: str,
    features_table: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads orderbook snapshots + features for a market/time range.

    Expected minimum key columns:
      - market_id
      - token_id
      - ts_utc
    But we merge later using only the keys that exist in BOTH tables, safely.
    """
    assert_table_exists(conn, snapshots_table)
    assert_table_exists(conn, features_table)

    where = "WHERE market_id = ?"
    params: List[Any] = [market_id]

    if start_utc:
        where += " AND ts_utc >= ?"
        params.append(start_utc)
    if end_utc:
        where += " AND ts_utc <= ?"
        params.append(end_utc)

    snap_sql = f"""
        SELECT *
        FROM {snapshots_table}
        {where}
        ORDER BY token_id, ts_utc
    """

    feat_sql = f"""
        SELECT *
        FROM {features_table}
        {where}
        ORDER BY token_id, ts_utc
    """

    try:
        snapshots = _read_sql_df(conn, snap_sql, tuple(params))
    except Exception as e:
        raise RuntimeError(
            f"Failed reading snapshots table '{snapshots_table}'. "
            f"Check table name/columns in read_tables(). Error: {e}"
        )

    try:
        feats = _read_sql_df(conn, feat_sql, tuple(params))
    except Exception as e:
        raise RuntimeError(
            f"Failed reading features table '{features_table}'. "
            f"Check table name/columns in read_tables(). Error: {e}"
        )

    return snapshots, feats


# -----------------------------
# JSON levels expansion
# -----------------------------
def _safe_json_loads(x: Any) -> List[Dict[str, Any]]:
    if x is None:
        return []
    if isinstance(x, (list, dict)):
        return x if isinstance(x, list) else []
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = json.loads(x)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def expand_levels(df: pd.DataFrame, col_json: str, side_prefix: str, max_levels: int = 10) -> pd.DataFrame:
    """
    Expand JSON levels like:
      [{"price":0.99,"size":14733.62}, {"price":0.98,"size":6152.16}, ...]

    into columns:
      bid_p1, bid_s1, bid_p2, bid_s2, ...
    """
    if col_json not in df.columns:
        return df

    levels = df[col_json].apply(_safe_json_loads)

    for i in range(1, max_levels + 1):
        df[f"{side_prefix}_p{i}"] = pd.NA
        df[f"{side_prefix}_s{i}"] = pd.NA

    def fill_row(row_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, lvl in enumerate(row_levels[:max_levels], start=1):
            out[f"{side_prefix}_p{i}"] = lvl.get("price")
            out[f"{side_prefix}_s{i}"] = lvl.get("size")
        return out

    expanded = levels.apply(fill_row).apply(pd.Series)
    for c in expanded.columns:
        df[c] = expanded[c]

    return df


# -----------------------------
# Filtering / flagging logic
# -----------------------------
def _append_reason(series: pd.Series, reason: str) -> pd.Series:
    """
    Append a reason string to existing reasons with '|' separator.
    """
    s = series.fillna("").astype(str)
    empty = s.str.len() == 0
    out = s.copy()
    out[empty] = reason
    out[~empty] = out[~empty] + "|" + reason
    return out


def enforce_1to1_match(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Flag duplicated keys after merge.
    (If either table has multiple rows per key, outer join can create duplicates.)
    """
    merged = merged.copy()
    if "flagged" not in merged.columns:
        merged["flagged"] = False
    if "flag_reason" not in merged.columns:
        merged["flag_reason"] = ""

    key_cols = [c for c in ["token_id", "ts_utc"] if c in merged.columns]
    if len(key_cols) < 2:
        return merged

    dup_mask = merged.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        merged.loc[dup_mask, "flagged"] = True
        merged.loc[dup_mask, "flag_reason"] = _append_reason(merged.loc[dup_mask, "flag_reason"], "duplicate_key_rows")
    return merged


def _infer_interval_seconds(ts: pd.Series) -> Optional[float]:
    """
    Infer typical interval for a token based on median time delta.
    ts must be ISO strings (ts_utc). We parse to datetime.
    """
    if len(ts) < 3:
        return None
    dt = pd.to_datetime(ts, utc=True, errors="coerce").dropna().sort_values()
    if len(dt) < 3:
        return None
    deltas = dt.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if len(deltas) < 2:
        return None
    return float(deltas.median())


def flag_by_token_times(cfg: Config, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows that look "corrupted" in time series sense:
    - bad timestamp parse
    - non-monotonic timestamps within token
    - irregular delta times outside tolerance
    - off-grid timestamps (optional, if grid_slack_s set)
    """
    merged = merged.copy()

    if "flagged" not in merged.columns:
        merged["flagged"] = False
    if "flag_reason" not in merged.columns:
        merged["flag_reason"] = ""

    if "token_id" not in merged.columns or "ts_utc" not in merged.columns:
        return merged

    merged["__dt"] = pd.to_datetime(merged["ts_utc"], utc=True, errors="coerce")

    bad_dt = merged["__dt"].isna()
    if bad_dt.any():
        merged.loc[bad_dt, "flagged"] = True
        merged.loc[bad_dt, "flag_reason"] = _append_reason(merged.loc[bad_dt, "flag_reason"], "bad_ts_parse")

    # Work token by token
    for token_id, gidx in merged.groupby("token_id").groups.items():
        sub = merged.loc[gidx].sort_values("__dt")
        if len(sub) < cfg.min_samples:
            continue

        dt = sub["__dt"]
        deltas = dt.diff().dt.total_seconds()

        nonpos = deltas <= 0
        if nonpos.any():
            bad_rows = sub.index[nonpos.fillna(False)]
            merged.loc[bad_rows, "flagged"] = True
            merged.loc[bad_rows, "flag_reason"] = _append_reason(
                merged.loc[bad_rows, "flag_reason"], "non_monotonic_ts"
            )

        expected = cfg.interval if cfg.interval is not None else _infer_interval_seconds(sub["ts_utc"])
        if expected and expected > 0:
            lo = expected * (1.0 - cfg.tolerance)
            hi = expected * (1.0 + cfg.tolerance)

            irregular = (deltas.notna()) & ((deltas < lo) | (deltas > hi))
            if irregular.any():
                bad_rows = sub.index[irregular]
                merged.loc[bad_rows, "flagged"] = True
                merged.loc[bad_rows, "flag_reason"] = _append_reason(
                    merged.loc[bad_rows, "flag_reason"], f"irregular_dt(expected~{expected:.2f}s)"
                )

            if cfg.grid_slack_s is not None:
                t0 = dt.iloc[0]
                sec = (dt - t0).dt.total_seconds()
                nearest = (sec / expected).round() * expected
                off = (sec - nearest).abs()
                offgrid = off > float(cfg.grid_slack_s)
                if offgrid.any():
                    bad_rows = sub.index[offgrid]
                    merged.loc[bad_rows, "flagged"] = True
                    merged.loc[bad_rows, "flag_reason"] = _append_reason(
                        merged.loc[bad_rows, "flag_reason"], f"off_grid(slack>{cfg.grid_slack_s}s)"
                    )

    merged.drop(columns=["__dt"], inplace=True, errors="ignore")
    return merged


def export_market_metadata(
    db_path: str,
    snapshots_table: str,
    features_table: str,
    out_path: str,
) -> None:
    """
    Export per-market counts so you can identify the largest markets for training.

    We compute:
      - snapshot_rows: rows in orderbook_snapshots per market_id
      - feature_rows : rows in features_orderbook per market_id
      - snapshot_tokens: distinct token_id count (if token_id exists)
      - snapshot_min_ts / snapshot_max_ts: coverage window (if ts_utc exists)

    Output: CSV sorted by snapshot_rows desc (primary signal for "largest").
    """
    conn = sqlite3.connect(db_path)
    try:
        assert_table_exists(conn, snapshots_table)
        assert_table_exists(conn, features_table)

        # ---- snapshots aggregation ----
        # token_id/ts_utc should exist in your schema; if not, we fallback gracefully.
        snap_cols = pd.read_sql_query(f"PRAGMA table_info({snapshots_table})", conn)["name"].tolist()
        has_token = "token_id" in snap_cols
        has_ts = "ts_utc" in snap_cols

        token_expr = "COUNT(DISTINCT token_id) AS snapshot_tokens" if has_token else "NULL AS snapshot_tokens"
        min_expr = "MIN(ts_utc) AS snapshot_min_ts" if has_ts else "NULL AS snapshot_min_ts"
        max_expr = "MAX(ts_utc) AS snapshot_max_ts" if has_ts else "NULL AS snapshot_max_ts"

        snap_sql = f"""
            SELECT
                market_id,
                COUNT(*) AS snapshot_rows,
                {token_expr},
                {min_expr},
                {max_expr}
            FROM {snapshots_table}
            GROUP BY market_id
        """
        snap_df = pd.read_sql_query(snap_sql, conn)

        # ---- features aggregation ----
        feat_sql = f"""
            SELECT
                market_id,
                COUNT(*) AS feature_rows
            FROM {features_table}
            GROUP BY market_id
        """
        feat_df = pd.read_sql_query(feat_sql, conn)

        # Outer merge so markets that appear in only one table still show up
        meta = snap_df.merge(feat_df, on="market_id", how="outer")

        # Fill NaNs for counts
        for c in ["snapshot_rows", "feature_rows", "snapshot_tokens"]:
            if c in meta.columns:
                meta[c] = meta[c].fillna(0).astype(int)

        # Sort: biggest snapshot_rows first (main training signal)
        meta = meta.sort_values(["snapshot_rows", "feature_rows"], ascending=[False, False])

        meta.to_csv(out_path, index=False)

        print("Market metadata exported.")
        print(f"  snapshots_table : {snapshots_table}")
        print(f"  features_table  : {features_table}")
        print(f"  markets         : {len(meta)}")
        print(f"  out             : {out_path}")

        # Print top 10 to console for convenience
        cols_to_show = [c for c in ["market_id", "snapshot_rows", "feature_rows", "snapshot_tokens"] if c in meta.columns]
        print("\nTop 10 markets by snapshot_rows:")
        print(meta[cols_to_show].head(10).to_string(index=False))

    finally:
        conn.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export merged Polymarket dataset for a market_id, with optional corrupted-row filtering."
    )
    ap.add_argument("--db", default="polymarket.db", help="SQLite DB path (default: polymarket.db)")
    ap.add_argument("--market-id", required=False, help="Market ID to export (required unless --export-market-metadata)")

    # Table names (defaults match your DB)
    ap.add_argument(
        "--snapshots-table",
        default="orderbook_snapshots",
        help="Snapshots table name (default: orderbook_snapshots)",
    )
    ap.add_argument(
        "--features-table",
        default="features_orderbook",
        help="Features table name (default: features_orderbook)",
    )

    ap.add_argument("--start", default=None, help='Start time (inclusive). E.g. "2026-02-01T00:00:00Z"')
    ap.add_argument("--end", default=None, help='End time (inclusive). E.g. "2026-02-10T00:00:00Z"')
    ap.add_argument("--tz", default="UTC", help='Timezone for naive start/end (default: UTC). E.g. "America/Chicago"')

    ap.add_argument("--out-clean", default="clean_orderbook_dataset.csv", help="Main CSV output")
    ap.add_argument("--out-flagged", default="flagged_corrupted_rows.csv", help="Flagged rows CSV output")

    ap.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Expected interval seconds. If omitted, inferred per token_id via median delta.",
    )
    ap.add_argument("--tolerance", type=float, default=0.35, help="Relative tolerance for delta checks (default ±35%)")
    ap.add_argument("--min-samples", type=int, default=8, help="Minimum rows per token_id before checks (default 8)")
    ap.add_argument("--grid-slack", type=float, default=None, help="Absolute seconds allowed off-grid (optional).")

    ap.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable corrupted-data filtering; export everything (useful for debugging).",
    )

    ap.add_argument("--max-levels", type=int, default=10, help="Max bid/ask levels to expand (default 10)")
    ap.add_argument(
        "--export-market-metadata",
        action="store_true",
        help="Export per-market row counts metadata and exit (does not require --market-id).",
    )
    ap.add_argument(
        "--market-metadata-out",
        default="market_metadata.csv",
        help="Output CSV for per-market metadata (default: market_metadata.csv)",
    )


    args = ap.parse_args()
    if not args.export_market_metadata and not args.market_id:
        ap.error("--market-id is required unless you pass --export-market-metadata")

    start_utc = parse_time_to_utc_iso(args.start, args.tz)
    end_utc = parse_time_to_utc_iso(args.end, args.tz)

    cfg = Config(
        db=args.db,
        market_id=str(args.market_id),
        start_utc=start_utc,
        end_utc=end_utc,
        out_clean=args.out_clean,
        out_flagged=args.out_flagged,
        interval=args.interval,
        tolerance=float(args.tolerance),
        min_samples=int(args.min_samples),
        grid_slack_s=args.grid_slack,
        enable_filter=(not args.no_filter),
        snapshots_table=args.snapshots_table,
        features_table=args.features_table,
    )

    if args.export_market_metadata:
        export_market_metadata(
            db_path=args.db,
            snapshots_table=args.snapshots_table,
            features_table=args.features_table,
            out_path=args.market_metadata_out,
        )
        return

    conn = sqlite3.connect(cfg.db)
    try:
        snapshots, feats = read_tables(
            conn,
            cfg.market_id,
            cfg.start_utc,
            cfg.end_utc,
            cfg.snapshots_table,
            cfg.features_table,
        )
    finally:
        conn.close()

    if len(snapshots) == 0 and len(feats) == 0:
        print("No rows found for that market/time range. Nothing to export.")
        return

    # Merge snapshots + features safely using only keys present in both
    if len(snapshots) > 0 and len(feats) > 0:
        candidate_keys = ["market_id", "token_id", "ts_utc"]
        keys = [k for k in candidate_keys if k in snapshots.columns and k in feats.columns]
        if not keys:
            raise RuntimeError(
                "No common merge keys found between snapshots and features.\n"
                f"snapshots cols={list(snapshots.columns)}\n"
                f"features  cols={list(feats.columns)}"
            )
        merged = snapshots.merge(
            feats,
            on=keys,
            how="outer",
            suffixes=("", "_feat"),
        )
    elif len(snapshots) > 0:
        merged = snapshots.copy()
    else:
        merged = feats.copy()

    # Quick load summary
    print("\n--- LOAD SUMMARY ---")
    print(f"snapshots table : {cfg.snapshots_table} rows={len(snapshots)}")
    print(f"features table  : {cfg.features_table} rows={len(feats)}")
    print(f"merged rows     : {len(merged)}")
    if "token_id" in merged.columns:
        print(f"tokens          : {merged['token_id'].nunique(dropna=True)}")
    print(f"filtering       : {'ON' if cfg.enable_filter else 'OFF (--no-filter)'}")
    print("--- END ---\n")

    # Filtering branch
    if cfg.enable_filter:
        merged = enforce_1to1_match(merged)
        merged = flag_by_token_times(cfg, merged)

        flagged = merged[merged["flagged"]].copy()
        flagged.sort_values([c for c in ["token_id", "ts_utc"] if c in flagged.columns], inplace=True, na_position="last")
        flagged.to_csv(cfg.out_flagged, index=False)

        out_df = merged[~merged["flagged"]].copy()
    else:
        out_df = merged.copy()
        flagged = pd.DataFrame()

    # Convenience metric if present
    if "spread" in out_df.columns and "mid_price" in out_df.columns:
        out_df["relative_spread"] = out_df["spread"] / out_df["mid_price"]

    # Expand orderbook JSON if present
    out_df = expand_levels(out_df, "bids_top_n_json", "bid", max_levels=int(args.max_levels))
    out_df = expand_levels(out_df, "asks_top_n_json", "ask", max_levels=int(args.max_levels))

    # Sort + export
    sort_cols = [c for c in ["market_id", "token_id", "ts_utc"] if c in out_df.columns]
    if sort_cols:
        out_df.sort_values(sort_cols, inplace=True, na_position="last")

    out_df.to_csv(cfg.out_clean, index=False)

    print("Done.")
    print(f"  market_id     : {cfg.market_id}")
    print(f"  start_utc     : {cfg.start_utc}")
    print(f"  end_utc       : {cfg.end_utc}")
    print(f"  exported rows : {len(out_df)} -> {cfg.out_clean}")

    if cfg.enable_filter:
        print(f"  flagged rows  : {len(flagged)} -> {cfg.out_flagged}")
        if len(flagged) > 0 and "flag_reason" in flagged.columns:
            print("\nTop flag reasons:")
            print(flagged["flag_reason"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
