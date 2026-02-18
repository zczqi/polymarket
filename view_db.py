
import argparse
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


# ------------------------
# Helpers for snapshots
# ------------------------

def load_levels(levels_json: Optional[str]) -> List[Dict[str, Any]]:
    if not levels_json:
        return []
    try:
        v = json.loads(levels_json)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def print_levels(title: str, levels: List[Dict[str, Any]], max_rows: int = 10) -> None:
    print(title)
    if not levels:
        print("  (empty)")
        return
    for lvl in levels[:max_rows]:
        print(f"  price={lvl.get('price')}  size={lvl.get('size')}")
    if len(levels) > max_rows:
        print(f"  ... ({len(levels)} levels total, showing {max_rows})")


def fetch_snapshots(
    conn: sqlite3.Connection,
    limit: Optional[int],
    token_id: Optional[str] = None
) -> List[Tuple[Any, ...]]:
    cur = conn.cursor()

    sql = """
    SELECT id, market_id, token_id, outcome, ts_utc,
           best_bid, best_ask, spread,
           bids_top_n_json, asks_top_n_json
    FROM orderbook_snapshots
    """

    params: List[Any] = []
    if token_id:
        sql += " WHERE token_id = ?"
        params.append(token_id)

    sql += " ORDER BY id DESC"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    cur.execute(sql, params)
    return cur.fetchall()


def fetch_feature_for_snapshot(
    conn: sqlite3.Connection,
    token_id: str,
    ts_utc: str
) -> Optional[Tuple[Any, ...]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id,
                   mid_price, microprice,
                   depth_bid_top5, depth_ask_top5, imbalance_top5,
                   hours_to_expiry
            FROM features_orderbook
            WHERE token_id = ? AND ts_utc = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (token_id, ts_utc),
        )
    except sqlite3.OperationalError:
        return None

    return cur.fetchone()


# ------------------------
# View markets (metadata)
# ------------------------

def view_markets(conn: sqlite3.Connection, limit: int) -> None:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT market_id,
                   market_type,
                   status,
                   end_time,
                   substr(question, 1, 80) AS question
            FROM markets
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
    except sqlite3.OperationalError:
        print("No markets table found. Run ingest_markets.py or ingest_collect_build.py first.")
        return

    rows = cur.fetchall()
    if not rows:
        print("markets table exists but is empty.")
        return

    print("=" * 80)
    print(f"Latest {len(rows)} markets")
    for r in rows:
        print(r)


# ------------------------
# View snapshots + features
# ------------------------

def view_snapshots_and_features(
    conn: sqlite3.Connection,
    limit: Optional[int],
    token_id: Optional[str],
    show_levels: int
) -> None:
    snapshots = fetch_snapshots(conn, limit=limit, token_id=token_id)

    if not snapshots:
        print("No snapshots found in orderbook_snapshots.")
        print("Run collect_orderbooks.py / collect_and_build.py / ingest_collect_build.py first.")
        return

    for (sid, market_id, tok_id, outcome, ts_utc,
         best_bid, best_ask, spread, bids_json, asks_json) in snapshots:

        print("\n" + "=" * 80)
        print(f"SNAPSHOT id={sid}")
        print(f"market_id: {market_id}")
        print(f"token_id : {tok_id}")
        print(f"outcome  : {outcome}")
        print(f"ts_utc   : {ts_utc}")
        print(f"best_bid : {best_bid}")
        print(f"best_ask : {best_ask}")
        print(f"spread   : {spread}")

        bids = load_levels(bids_json)
        asks = load_levels(asks_json)

        print_levels("\n--- TOP BIDS ---", bids, max_rows=show_levels)
        print_levels("\n--- TOP ASKS ---", asks, max_rows=show_levels)

        feat = fetch_feature_for_snapshot(conn, token_id=tok_id, ts_utc=ts_utc)
        print("\n--- FEATURES (matched by token_id + ts_utc) ---")
        if feat is None:
            print("  (features_orderbook table not found, or no features for this snapshot)")
        else:
            (fid, mid_price, microprice, depth_bid5, depth_ask5, imb5, hours_to_expiry) = feat
            print(f"  feature_row_id   : {fid}")
            print(f"  mid_price        : {mid_price}")
            print(f"  microprice       : {microprice}")
            print(f"  depth_bid_top5   : {depth_bid5}")
            print(f"  depth_ask_top5   : {depth_ask5}")
            print(f"  imbalance_top5   : {imb5}")
            print(f"  hours_to_expiry  : {hours_to_expiry}")


# ------------------------
# CLI
# ------------------------

def main():
    p = argparse.ArgumentParser(description="View Polymarket SQLite DB contents")
    p.add_argument("--db", default="polymarket.db", help="SQLite DB path (default: polymarket.db)")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_mk = sub.add_parser("markets", help="Show latest markets metadata")
    p_mk.add_argument("--limit", type=int, default=10, help="How many latest markets to show (default 10)")

    p_sn = sub.add_parser("snapshots", help="Show orderbook snapshots + features")
    p_sn.add_argument("--limit", type=int, default=None, help="How many latest snapshots to show (default: all)")
    p_sn.add_argument("--token-id", default=None, help="Optional: only show snapshots for one token_id")
    p_sn.add_argument("--show-levels", type=int, default=10, help="How many bid/ask levels to print (default 10)")

    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.cmd == "markets":
            view_markets(conn, limit=args.limit)
        elif args.cmd == "snapshots":
            view_snapshots_and_features(
                conn,
                limit=args.limit,
                token_id=args.token_id,
                show_levels=args.show_levels
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
