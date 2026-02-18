#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-stop script:
1) Ingest ONE event (by slug) from Gamma into SQLite markets table
2) Collect CLOB orderbook snapshots for that event
3) Compute and store orderbook-based features per snapshot

Usage:
  python ingest_collect_build.py --event-slug us-strikes-iran-by --interval 60 --depth 30 --iterations 10
"""

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


# ============================================================
# Helpers (time / json)
# ============================================================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    s = str(dt_str).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def safe_json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def parse_jsonish_list(x: Any) -> List[Any]:
    """
    Gamma sometimes returns arrays as real lists or JSON strings.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                return v if isinstance(v, list) else []
            except json.JSONDecodeError:
                return []
        return [p.strip() for p in s.split(",") if p.strip()]
    return []


# ============================================================
# Gamma ingest: event -> markets -> upsert into `markets`
# ============================================================

MARKETS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
  market_id TEXT PRIMARY KEY,
  question TEXT,
  market_type TEXT,          -- 'binary' | 'multi' | 'unknown'
  end_time TEXT,             -- ISO time string from Gamma endDate
  status TEXT,               -- 'active' | 'closed' | 'archived' | 'unknown'
  rules_text TEXT,           -- Gamma description (rules)
  resolution_source TEXT,    -- Gamma resolutionSource
  updated_at TEXT,           -- ingestion timestamp (UTC ISO)
  raw_json TEXT              -- raw Gamma market object
);

CREATE INDEX IF NOT EXISTS idx_markets_end_time ON markets(end_time);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);
"""

MARKETS_UPSERT_SQL = """
INSERT INTO markets (
  market_id, question, market_type, end_time, status,
  rules_text, resolution_source, updated_at, raw_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(market_id) DO UPDATE SET
  question=excluded.question,
  market_type=excluded.market_type,
  end_time=excluded.end_time,
  status=excluded.status,
  rules_text=excluded.rules_text,
  resolution_source=excluded.resolution_source,
  updated_at=excluded.updated_at,
  raw_json=excluded.raw_json;
"""

def derive_market_type_from_outcomes(outcomes: List[Any]) -> str:
    if len(outcomes) == 2:
        return "binary"
    if len(outcomes) > 2:
        return "multi"
    return "unknown"

def derive_status(market: Dict[str, Any]) -> str:
    archived = bool(market.get("archived"))
    closed = bool(market.get("closed"))
    active = bool(market.get("active"))
    if archived:
        return "archived"
    if closed and not active:
        return "closed"
    if active and not closed:
        return "active"
    return "unknown"

def fetch_event_by_slug(slug: str) -> Dict[str, Any]:
    r = requests.get(f"{GAMMA_BASE}/events/slug/{slug}", timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_markets_by_event_id(event_id: Any) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"event_id": event_id, "limit": 500, "offset": 0},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("Expected list from Gamma /markets?event_id=...")
    return data

def upsert_market(conn: sqlite3.Connection, row: Tuple[Any, ...]) -> None:
    conn.execute(MARKETS_UPSERT_SQL, row)

def ingest_event_markets(conn: sqlite3.Connection, event_slug: str) -> List[Dict[str, Any]]:
    """
    Fetch event by slug, obtain its markets list (embedded or via /markets?event_id),
    and upsert into `markets` table.

    Returns the market objects (list of dicts) for downstream token extraction.
    """
    conn.executescript(MARKETS_SCHEMA_SQL)
    conn.commit()

    event = fetch_event_by_slug(event_slug)

    markets = event.get("markets")
    if not isinstance(markets, list) or len(markets) == 0:
        event_id = event.get("id")
        if event_id is None:
            raise RuntimeError("Event response has no markets and no id.")
        markets = fetch_markets_by_event_id(event_id)

    ts = now_utc_iso()
    n = 0
    for m in markets:
        market_id = m.get("id")
        if market_id is None:
            continue

        outcomes = parse_jsonish_list(m.get("outcomes"))
        row = (
            str(market_id),
            m.get("question"),
            derive_market_type_from_outcomes(outcomes),
            m.get("endDate"),
            derive_status(m),
            (str(m.get("description")).strip() if m.get("description") else None),
            m.get("resolutionSource"),
            ts,
            safe_json_dumps(m),
        )
        upsert_market(conn, row)
        n += 1

    conn.commit()
    print(f"[ingest] Upserted {n} markets for event '{event_slug}' into markets table.")
    return markets


# ============================================================
# Orderbooks + features: schema + compute
# ============================================================

OBS_AND_FEATURES_SCHEMA = """
-- orderbook snapshots
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  market_id TEXT NOT NULL,
  token_id TEXT NOT NULL,
  outcome TEXT,
  ts_utc TEXT NOT NULL,

  best_bid REAL,
  best_ask REAL,
  spread REAL,

  bids_top_n_json TEXT,
  asks_top_n_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_obs_market_ts ON orderbook_snapshots(market_id, ts_utc);
CREATE INDEX IF NOT EXISTS idx_obs_token_ts  ON orderbook_snapshots(token_id, ts_utc);

-- features computed from snapshots
CREATE TABLE IF NOT EXISTS features_orderbook (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  token_id TEXT NOT NULL,
  market_id TEXT NOT NULL,
  outcome TEXT,
  ts_utc TEXT NOT NULL,

  best_bid REAL,
  best_ask REAL,
  bid_size_1 REAL,
  ask_size_1 REAL,

  mid_price REAL,
  microprice REAL,

  depth_bid_top5 REAL,
  depth_ask_top5 REAL,
  imbalance_top5 REAL,

  seconds_to_expiry REAL,
  hours_to_expiry REAL
);

CREATE INDEX IF NOT EXISTS idx_feat_token_ts ON features_orderbook(token_id, ts_utc);
"""

def init_obs_and_features(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(OBS_AND_FEATURES_SCHEMA)
    conn.commit()

def insert_snapshot(
    conn: sqlite3.Connection,
    market_id: str,
    token_id: str,
    outcome: Optional[str],
    ts_utc: str,
    best_bid: Optional[float],
    best_ask: Optional[float],
    spread: Optional[float],
    bids_top_n: List[Dict[str, Any]],
    asks_top_n: List[Dict[str, Any]],
) -> None:
    conn.execute(
        """
        INSERT INTO orderbook_snapshots (
          market_id, token_id, outcome, ts_utc,
          best_bid, best_ask, spread,
          bids_top_n_json, asks_top_n_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            market_id,
            token_id,
            outcome,
            ts_utc,
            best_bid,
            best_ask,
            spread,
            safe_json_dumps(bids_top_n),
            safe_json_dumps(asks_top_n),
        ),
    )

def insert_feature_row(conn: sqlite3.Connection, row: Tuple[Any, ...]) -> None:
    conn.execute(
        """
        INSERT INTO features_orderbook (
          token_id, market_id, outcome, ts_utc,
          best_bid, best_ask, bid_size_1, ask_size_1,
          mid_price, microprice,
          depth_bid_top5, depth_ask_top5, imbalance_top5,
          seconds_to_expiry, hours_to_expiry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )

# orderbook helpers

def level_price_size(level: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    p = level.get("price")
    s = level.get("size")
    try:
        p_f = float(p) if p is not None else None
    except Exception:
        p_f = None
    try:
        s_f = float(s) if s is not None else None
    except Exception:
        s_f = None
    return p_f, s_f

def best_price_and_size(levels: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not levels:
        return None, None
    return level_price_size(levels[0])

def sum_top_n_sizes(levels: List[Dict[str, Any]], n: int) -> float:
    total = 0.0
    for lvl in levels[:n]:
        _, sz = level_price_size(lvl)
        if sz is not None:
            total += sz
    return total

# features

def mid_price(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0

def microprice(
    bid: Optional[float], ask: Optional[float],
    bid_size: Optional[float], ask_size: Optional[float]
) -> Optional[float]:
    # microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    if bid is None or ask is None or bid_size is None or ask_size is None:
        return None
    denom = bid_size + ask_size
    if denom == 0:
        return None
    return (ask * bid_size + bid * ask_size) / denom

def imbalance(depth_bid: float, depth_ask: float) -> Optional[float]:
    denom = depth_bid + depth_ask
    if denom == 0:
        return None
    return (depth_bid - depth_ask) / denom

# CLOB fetch

def fetch_books(token_ids: List[str]) -> List[Dict[str, Any]]:
    payload = [{"token_id": tid} for tid in token_ids]
    r = requests.post(f"{CLOB_BASE}/books", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError("Expected list from CLOB POST /books")
    return data

def normalize_book_response(book_obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    token_id = str(book_obj.get("token_id") or book_obj.get("tokenId") or book_obj.get("asset_id") or "")
    bids = book_obj.get("bids") or []
    asks = book_obj.get("asks") or []
    if not isinstance(bids, list): bids = []
    if not isinstance(asks, list): asks = []
    return token_id, bids, asks


# ============================================================
# Token extraction (from ingested markets)
# ============================================================

def extract_market_tokens(market: Dict[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    """
    Return list of (market_id, token_id, outcome_label) using:
      - clobTokenIds
      - outcomes
    """
    market_id = str(market.get("id"))
    token_ids = parse_jsonish_list(market.get("clobTokenIds"))
    outcomes = parse_jsonish_list(market.get("outcomes"))

    out: List[Tuple[str, str, Optional[str]]] = []
    for i, tid in enumerate(token_ids):
        label = str(outcomes[i]) if i < len(outcomes) else None
        out.append((market_id, str(tid), label))
    return out


def build_market_endtime_map(conn: sqlite3.Connection) -> Dict[str, Optional[str]]:
    cur = conn.cursor()
    cur.execute("SELECT market_id, end_time FROM markets")
    return {str(mid): et for (mid, et) in cur.fetchall()}


# ============================================================
# Main run loop
# ============================================================

def run(
    db_path: str,
    event_slug: str,
    depth_n: int,
    interval_s: float,
    iterations: Optional[int],
) -> None:
    conn = sqlite3.connect(db_path)

    # 1) Ingest markets metadata for this event
    markets = ingest_event_markets(conn, event_slug)

    # 2) Prepare snapshot+feature tables
    init_obs_and_features(conn)

    # 3) Market end_time mapping for time-to-expiry
    market_endtime = build_market_endtime_map(conn)

    # 4) Extract tokens for CLOB orderbooks
    market_tokens: List[Tuple[str, str, Optional[str]]] = []
    for m in markets:
        market_tokens.extend(extract_market_tokens(m))

    if not market_tokens:
        raise RuntimeError("No clobTokenIds found; cannot collect orderbooks for this event.")

    token_to_market: Dict[str, str] = {}
    token_to_outcome: Dict[str, Optional[str]] = {}
    for market_id, token_id, outcome in market_tokens:
        token_to_market[token_id] = market_id
        token_to_outcome[token_id] = outcome

    token_ids = list(token_to_market.keys())

    print(f"[collect] Event '{event_slug}': markets={len(markets)}, tokens={len(token_ids)}")
    print("[collect] Collecting snapshots + building features... Ctrl+C to stop.")

    i = 0
    try:
        while True:
            ts = now_utc_iso()

            books = fetch_books(token_ids)

            written = 0
            for b in books:
                tid, bids, asks = normalize_book_response(b)
                if not tid:
                    continue

                bids_top = bids[:depth_n]
                asks_top = asks[:depth_n]

                bb_price, bb_size = best_price_and_size(bids)
                ba_price, ba_size = best_price_and_size(asks)

                best_bid_val = bb_price
                best_ask_val = ba_price
                spread_val = (best_ask_val - best_bid_val) if (best_bid_val is not None and best_ask_val is not None) else None

                market_id = token_to_market.get(tid, "unknown")
                outcome = token_to_outcome.get(tid)

                # A) snapshot row
                insert_snapshot(
                    conn=conn,
                    market_id=market_id,
                    token_id=tid,
                    outcome=outcome,
                    ts_utc=ts,
                    best_bid=best_bid_val,
                    best_ask=best_ask_val,
                    spread=spread_val,
                    bids_top_n=bids_top,
                    asks_top_n=asks_top,
                )

                # B) feature row
                mp = mid_price(best_bid_val, best_ask_val)
                mic = microprice(best_bid_val, best_ask_val, bb_size, ba_size)

                depth_bid5 = sum_top_n_sizes(bids, 5)
                depth_ask5 = sum_top_n_sizes(asks, 5)
                imb5 = imbalance(depth_bid5, depth_ask5)

                end_time = market_endtime.get(str(market_id))
                ts_dt = parse_iso(ts)
                end_dt = parse_iso(end_time) if end_time else None
                seconds_to_expiry = None
                hours_to_expiry = None
                if ts_dt and end_dt:
                    delta = (end_dt - ts_dt).total_seconds()
                    seconds_to_expiry = delta
                    hours_to_expiry = delta / 3600.0

                feat_row = (
                    tid, market_id, outcome, ts,
                    best_bid_val, best_ask_val, bb_size, ba_size,
                    mp, mic,
                    depth_bid5, depth_ask5, imb5,
                    seconds_to_expiry, hours_to_expiry
                )
                insert_feature_row(conn, feat_row)

                written += 1

            conn.commit()

            i += 1
            print(f"[{ts}] wrote {written} tokens (iteration {i})")

            if iterations is not None and i >= iterations:
                break

            time.sleep(interval_s)

    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser(description="Ingest event markets + collect orderbooks + build features")
    p.add_argument("--db", default="polymarket.db")
    p.add_argument("--event-slug", required=True)
    p.add_argument("--depth", type=int, default=30, help="Store top N levels (default 30)")
    p.add_argument("--interval", type=float, default=60.0, help="Seconds between snapshots (default 60)")
    p.add_argument("--iterations", type=int, default=None, help="Stop after N iterations (default: run forever)")
    args = p.parse_args()

    run(
        db_path=args.db,
        event_slug=args.event_slug,
        depth_n=args.depth,
        interval_s=args.interval,
        iterations=args.iterations,
    )

if __name__ == "__main__":
    main()
