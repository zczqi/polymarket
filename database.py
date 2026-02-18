#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket Gamma -> SQLite

This file supports TWO modes:

1) Bulk ingest (original behavior)
   Examples:
     python database.py --db polymarket.db --limit 100 --closed false
     python database.py --db polymarket.db --limit 100 --closed true --max-pages 5

2) Track top-N by "session" (sticky watchlist until markets end)
   Examples:
     python database.py --mode track --db polymarket.db --session trending --top 10
     python database.py --mode track --db polymarket.db --session trending,culture,tech --top 20 --refresh-every 120

Session behavior:
- Special sessions:
    trending -> order by volume24hr desc
    new      -> order by createdAt desc
- Any other value is treated as a Gamma "tag" and resolved by:
    - exact slug via GET /tags/slug/{slug}
    - fallback: label/partial match via GET /tags (with suggestions if ambiguous)
"""

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


# ----------------------------
# Utilities
# ----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_bool_str(v: bool) -> str:
    return "true" if v else "false"


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_session_list(s: str) -> List[str]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    out: List[str] = []
    seen = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


# ----------------------------
# Parsing / normalization
# ----------------------------

def parse_outcomes(outcomes_field: Any) -> List[str]:
    if outcomes_field is None:
        return []
    if isinstance(outcomes_field, list):
        return [str(x) for x in outcomes_field]
    if isinstance(outcomes_field, str):
        s = outcomes_field.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return [str(x) for x in v]
            except json.JSONDecodeError:
                pass
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]
    return []


def derive_market_type(outcomes: List[str]) -> str:
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


def extract_rules_text(market: Dict[str, Any]) -> Optional[str]:
    rules = market.get("description")
    if rules is None:
        return None
    rules = str(rules).strip()
    return rules if rules else None


# ----------------------------
# SQLite storage
# ----------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS markets (
  market_id TEXT PRIMARY KEY,
  question TEXT,
  market_type TEXT,
  end_time TEXT,
  status TEXT,
  rules_text TEXT,
  resolution_source TEXT,
  updated_at TEXT,
  raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_markets_end_time ON markets(end_time);
CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status);

-- Sticky watchlist: once a market appears in a session's top N, track until ended
CREATE TABLE IF NOT EXISTS tracked_markets (
  market_id TEXT PRIMARY KEY,
  sessions TEXT,             -- comma-separated sessions that have seen it
  first_seen_at TEXT,
  last_seen_at TEXT,
  ended INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tracked_ended ON tracked_markets(ended);
"""

UPSERT_MARKET_SQL = """
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

UPSERT_TRACKED_SQL = """
INSERT INTO tracked_markets (
  market_id, sessions, first_seen_at, last_seen_at, ended
) VALUES (?, ?, ?, ?, 0)
ON CONFLICT(market_id) DO UPDATE SET
  sessions=excluded.sessions,
  last_seen_at=excluded.last_seen_at;
"""


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def upsert_market_row(conn: sqlite3.Connection, row: Tuple[Any, ...]) -> None:
    conn.execute(UPSERT_MARKET_SQL, row)


def get_tracked_sessions(conn: sqlite3.Connection, market_id: str) -> str:
    cur = conn.execute("SELECT sessions FROM tracked_markets WHERE market_id=?", (market_id,))
    r = cur.fetchone()
    return r[0] if r and r[0] else ""


def merge_sessions(existing: str, new_session: str) -> str:
    existing_set = {p.strip().lower() for p in existing.split(",") if p.strip()}
    existing_set.add(new_session.lower())
    return ",".join(sorted(existing_set))


def upsert_tracked_market(conn: sqlite3.Connection, market_id: str, session_name: str, ts: str) -> None:
    existing = get_tracked_sessions(conn, market_id)
    merged = merge_sessions(existing, session_name) if existing else session_name.lower()
    conn.execute(UPSERT_TRACKED_SQL, (market_id, merged, ts, ts))


def mark_tracked_ended(conn: sqlite3.Connection, market_id: str) -> None:
    conn.execute("UPDATE tracked_markets SET ended=1 WHERE market_id=?", (market_id,))


def list_active_tracked_ids(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT market_id FROM tracked_markets WHERE ended=0")
    return [r[0] for r in cur.fetchall()]


# ----------------------------
# HTTP client with retries
# ----------------------------

def make_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "polymarket-mispricing-capstone/1.3 (db+tracker)"})
    return session


# ----------------------------
# Gamma API
# ----------------------------

def gamma_get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: int = 30,
) -> Any:
    resp = session.get(url, params=params, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gamma API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def fetch_markets(
    session: requests.Session,
    *,
    limit: int,
    offset: int,
    closed: bool,
    order: Optional[str] = None,
    ascending: bool = False,
    tag_id: Optional[int] = None,
    timeout_s: int = 30,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit, "offset": offset, "closed": to_bool_str(closed)}
    if order is not None:
        params["order"] = order
        params["ascending"] = to_bool_str(ascending)
    if tag_id is not None:
        params["tag_id"] = str(tag_id)

    url = f"{GAMMA_BASE_URL}/markets"
    data = gamma_get_json(session, url, params=params, timeout_s=timeout_s)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response type from /markets: {type(data)}")
    return data


def fetch_market_by_id(session: requests.Session, market_id: str, timeout_s: int = 30) -> Dict[str, Any]:
    url = f"{GAMMA_BASE_URL}/markets/{market_id}"
    data = gamma_get_json(session, url, params=None, timeout_s=timeout_s)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type from /markets/{{id}}: {type(data)}")
    return data


_ALL_TAGS_CACHE: Optional[List[Dict[str, Any]]] = None


def list_all_tags(session: requests.Session, timeout_s: int = 30, page_limit: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch all tags from GET /tags (paged) and cache them in-memory.
    """
    global _ALL_TAGS_CACHE
    if _ALL_TAGS_CACHE is not None:
        return _ALL_TAGS_CACHE

    tags: List[Dict[str, Any]] = []
    offset = 0
    while True:
        batch = gamma_get_json(
            session,
            f"{GAMMA_BASE_URL}/tags",
            params={"limit": page_limit, "offset": offset},
            timeout_s=timeout_s,
        )
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected response type from /tags: {type(batch)}")
        if not batch:
            break
        tags.extend(batch)
        offset += page_limit

    _ALL_TAGS_CACHE = tags
    return tags


def fetch_tag_id_by_slug(session: requests.Session, slug: str, timeout_s: int = 30) -> int:
    """
    Resolve a tag id. Accepts:
      - exact slug (fast path via GET /tags/slug/{slug})
      - label or partial match fallback via GET /tags (so user can type 'tech' and still work)
    """
    key = (slug or "").strip()
    if not key:
        raise RuntimeError("Empty tag slug/label")

    # Fast path: exact slug lookup
    url = f"{GAMMA_BASE_URL}/tags/slug/{key}"
    resp = session.get(url, timeout=timeout_s)
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "id" in data:
            return int(data["id"])
        raise RuntimeError(f"Unexpected response from /tags/slug/{key}: {type(data)}")

    # If it's not a 404, surface the real error
    if resp.status_code != 404:
        raise RuntimeError(f"Gamma API error {resp.status_code}: {resp.text[:300]}")

    # Fallback: fetch tags and try to match by slug/label
    tags = list_all_tags(session, timeout_s=timeout_s)
    key_l = key.lower()

    def slug_of(t: Dict[str, Any]) -> str:
        return (t.get("slug") or "").strip()

    def label_of(t: Dict[str, Any]) -> str:
        return (t.get("label") or "").strip()

    # 1) exact matches
    exact = [t for t in tags if slug_of(t).lower() == key_l] or [t for t in tags if label_of(t).lower() == key_l]
    if len(exact) == 1 and "id" in exact[0]:
        return int(exact[0]["id"])

    # 2) partial matches
    candidates = [t for t in tags if key_l in slug_of(t).lower() or key_l in label_of(t).lower()]

    # Prefer “starts with” matches if that narrows it down
    starts = [t for t in candidates if slug_of(t).lower().startswith(key_l) or label_of(t).lower().startswith(key_l)]
    if len(starts) == 1 and "id" in starts[0]:
        return int(starts[0]["id"])
    if len(candidates) == 1 and "id" in candidates[0]:
        return int(candidates[0]["id"])

    # Build suggestions
    suggestions = [(label_of(t), slug_of(t)) for t in (starts or candidates) if slug_of(t)]
    suggestions = suggestions[:15]

    if suggestions:
        sug_txt = ", ".join([f"{lab or '(no label)'} -> {sl}" for lab, sl in suggestions])
        raise RuntimeError(
            f"Tag '{key}' not found as a slug. Possible matches: {sug_txt}. "
            f"Use the exact slug from GET /tags."
        )

    raise RuntimeError(
        f"Tag '{key}' not found. List available tags via GET /tags and use the tag's 'slug'."
    )


# ----------------------------
# Convert Gamma market -> row
# ----------------------------

def market_to_row(market: Dict[str, Any], ingested_at: str) -> Optional[Tuple[Any, ...]]:
    market_id = market.get("id")
    if market_id is None:
        return None

    outcomes = parse_outcomes(market.get("outcomes"))
    return (
        str(market_id),
        market.get("question"),
        derive_market_type(outcomes),
        market.get("endDate"),
        derive_status(market),
        extract_rules_text(market),
        market.get("resolutionSource"),
        ingested_at,
        safe_json_dumps(market),
    )


# ----------------------------
# Mode 1: Bulk ingest (original)
# ----------------------------

@dataclass
class IngestConfig:
    db_path: str
    limit: int
    closed: bool
    max_pages: Optional[int]
    sleep_s: float


def run_ingest(config: IngestConfig) -> None:
    conn = init_db(config.db_path)
    session = make_session()

    offset = 0
    pages = 0
    rows_upserted = 0

    try:
        while True:
            page = fetch_markets(
                session=session,
                limit=config.limit,
                offset=offset,
                closed=config.closed,
            )

            if not page:
                break

            ingested_at = now_utc_iso()

            for market in page:
                row = market_to_row(market, ingested_at)
                if row is None:
                    continue
                upsert_market_row(conn, row)
                rows_upserted += 1

            conn.commit()
            pages += 1
            offset += config.limit

            if config.max_pages is not None and pages >= config.max_pages:
                break

            if config.sleep_s > 0:
                time.sleep(config.sleep_s)

    finally:
        conn.close()

    print(f"Done (ingest). pages={pages}, rows_upserted={rows_upserted}, db={config.db_path}")


# ----------------------------
# Mode 2: Track top-N by session (sticky until ended)
# ----------------------------

SPECIAL_SESSIONS = {"trending", "new"}


def session_query_params(
    session_name: str,
    tag_id_cache: Dict[str, int],
    http: requests.Session,
) -> Dict[str, Any]:
    s = session_name.lower()

    if s == "trending":
        return {"order": "volume24hr", "ascending": False, "tag_id": None}
    if s == "new":
        return {"order": "createdAt", "ascending": False, "tag_id": None}

    # Default: treat as a Gamma tag slug/label and resolve to tag_id
    if s not in tag_id_cache:
        tag_id_cache[s] = fetch_tag_id_by_slug(http, s)
    return {"order": "volumeNum", "ascending": False, "tag_id": tag_id_cache[s]}


@dataclass
class TrackConfig:
    db_path: str
    sessions: List[str]
    top_n: int
    discover_closed: bool
    refresh_every_s: float
    per_market_sleep_s: float
    max_cycles: Optional[int]


def discover_topN(conn: sqlite3.Connection, http: requests.Session, cfg: TrackConfig, tag_id_cache: Dict[str, int]) -> int:
    discovered = 0
    ts = now_utc_iso()

    for session_name in cfg.sessions:
        qp = session_query_params(session_name, tag_id_cache, http)

        top_list = fetch_markets(
            http,
            limit=cfg.top_n,
            offset=0,
            closed=cfg.discover_closed,
            order=qp["order"],
            ascending=qp["ascending"],
            tag_id=qp["tag_id"],
        )

        for m in top_list:
            row = market_to_row(m, ts)
            if row is None:
                continue
            upsert_market_row(conn, row)
            upsert_tracked_market(conn, str(m["id"]), session_name, ts)
            discovered += 1

    conn.commit()
    return discovered


def follow_tracked_once(conn: sqlite3.Connection, http: requests.Session, cfg: TrackConfig) -> Tuple[int, int]:
    updated = 0
    ended_now = 0

    active_ids = list_active_tracked_ids(conn)
    for mid in active_ids:
        ts = now_utc_iso()
        m = fetch_market_by_id(http, mid)

        row = market_to_row(m, ts)
        if row is not None:
            upsert_market_row(conn, row)
            updated += 1

        status = derive_status(m)
        if status in ("closed", "archived"):
            mark_tracked_ended(conn, mid)
            ended_now += 1

        if cfg.per_market_sleep_s > 0:
            time.sleep(cfg.per_market_sleep_s)

    conn.commit()
    return updated, ended_now


def run_tracker(cfg: TrackConfig) -> None:
    conn = init_db(cfg.db_path)
    http = make_session()
    tag_id_cache: Dict[str, int] = {}

    cycle = 0
    try:
        while True:
            cycle += 1
            started = now_utc_iso()

            discovered = discover_topN(conn, http, cfg, tag_id_cache)
            updated, ended_now = follow_tracked_once(conn, http, cfg)
            active_remaining = len(list_active_tracked_ids(conn))

            print(
                f"[cycle {cycle}] {started} sessions={cfg.sessions} top={cfg.top_n} "
                f"discovered={discovered} updated={updated} ended_now={ended_now} active_tracked={active_remaining}"
            )

            if cfg.max_cycles is not None and cycle >= cfg.max_cycles:
                break
            if cfg.refresh_every_s <= 0:
                break

            time.sleep(cfg.refresh_every_s)

    finally:
        conn.close()

    print(f"Done (track). db={cfg.db_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Polymarket Gamma -> SQLite (ingest or track top-N).")

    p.add_argument(
        "--mode",
        type=str,
        default="ingest",
        choices=["ingest", "track"],
        help="Run mode: ingest (bulk) or track (top-N sticky watchlist). Default: ingest.",
    )

    p.add_argument("--db", dest="db_path", default="polymarket.db", help="SQLite DB path (default: polymarket.db)")

    # Ingest args (original)
    p.add_argument("--limit", type=int, default=100, help="Page size for bulk ingest pagination (default: 100)")
    p.add_argument("--closed", type=str, default="false", help="Bulk ingest closed markets? true/false (default: false)")
    p.add_argument("--max-pages", type=int, default=None, help="Bulk ingest: max pages to fetch (default: no limit)")
    p.add_argument("--sleep", dest="sleep_s", type=float, default=0.2, help="Bulk ingest: sleep between pages (default: 0.2)")

    # Track args
    p.add_argument(
        "--session",
        type=str,
        default="trending",
        help="Track mode: sessions to track (comma-separated). Special: trending,new. Otherwise any Gamma tag slug/label (e.g., culture, tech, politics, sports, crypto, ...).",
    )
    p.add_argument("--top", dest="top_n", type=int, default=10, help="Track mode: top N to track (default: 10)")
    p.add_argument(
        "--discover-closed",
        type=str,
        default="false",
        help="Track mode: include closed markets in discovery lists? true/false (default: false)",
    )
    p.add_argument(
        "--refresh-every",
        dest="refresh_every_s",
        type=float,
        default=300.0,
        help="Track mode: seconds between cycles. Set <=0 for one cycle only. (default: 300)",
    )
    p.add_argument(
        "--per-market-sleep",
        dest="per_market_sleep_s",
        type=float,
        default=0.1,
        help="Track mode: sleep between /markets/{id} calls (default: 0.1)",
    )
    p.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Track mode: stop after N cycles (default: run until Ctrl+C or refresh-every<=0)",
    )

    args = p.parse_args()

    if args.mode == "ingest":
        closed = args.closed.strip().lower() in ("true", "1", "yes", "y")
        return ("ingest", IngestConfig(
            db_path=args.db_path,
            limit=args.limit,
            closed=closed,
            max_pages=args.max_pages,
            sleep_s=args.sleep_s,
        ))

    # mode == track
    sessions = normalize_session_list(args.session)
    if not sessions:
        raise SystemExit("--session must contain at least one value")
    if args.top_n <= 0:
        raise SystemExit("--top must be > 0")
    discover_closed = args.discover_closed.strip().lower() in ("true", "1", "yes", "y")

    return ("track", TrackConfig(
        db_path=args.db_path,
        sessions=sessions,
        top_n=args.top_n,
        discover_closed=discover_closed,
        refresh_every_s=float(args.refresh_every_s),
        per_market_sleep_s=float(args.per_market_sleep_s),
        max_cycles=args.max_cycles,
    ))


def main():
    mode, cfg = parse_args()
    if mode == "ingest":
        run_ingest(cfg)
    else:
        run_tracker(cfg)


if __name__ == "__main__":
    main()