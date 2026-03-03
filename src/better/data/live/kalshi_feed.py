"""Kalshi prediction market WebSocket feed.

Connects to Kalshi's real-time WebSocket API to receive live MLB game
contract prices, converts them to fair probabilities, and stores them
in the existing odds_snapshots table (bookmaker='kalshi').

Architecture:
    KalshiAuthClient      — REST login → JWT token, automatic re-auth
    KalshiMarketDiscovery — finds today's MLB market tickers via REST
    KalshiWebSocketFeed   — async WebSocket listener, stores to DB

Usage::

    feed = await start_kalshi_feed()    # None if no credentials set
    if feed:
        asyncio.create_task(feed.run())
        ...
        feed.stop()

Kalshi WebSocket protocol:
    wss://trading-api.kalshi.com/trade-api/ws/v2

    Send: {"id": 1, "cmd": "login",     "params": {"token": "JWT"}}
    Send: {"id": 2, "cmd": "subscribe", "params": {
               "channels": ["ticker"],
               "market_tickers": ["MLBG-BOS-20260301-NYY"]}}
    Recv: {"type": "ticker", "msg": {
               "market_ticker": "MLBG-BOS-20260301-NYY",
               "yes_bid": 58, "yes_ask": 62,
               "last_price": 60, "volume": 1234}}

Price conversion:
    mid          = (yes_bid + yes_ask) / 2     # cents, 0-100
    home_fair    = mid / 100                   # decimal probability
    away_fair    = 1 - home_fair
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import httpx
import websockets

from better.config import settings
from better.data.db import get_connection
from better.utils.logging import get_logger
from better.utils.stats import implied_prob_to_american

log = get_logger(__name__)

# ── Kalshi team abbreviation → our 3-letter MLB codes ────────────────────
# Kalshi generally uses standard MLB abbreviations, but a few differ.
KALSHI_TO_MLB: dict[str, str] = {
    # Standard matches (identity mapping for the common ones)
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CWS", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC":  "KC",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "PHI": "PHI", "PIT": "PIT", "SD":  "SD",  "SF":  "SF",
    "SEA": "SEA", "STL": "STL", "TB":  "TB",  "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH",
    # Kalshi may use alternate codes
    "CHW": "CWS",   # Chicago White Sox alternate
    "KCR": "KC",    # Kansas City Royals alternate
    "SDP": "SD",    # San Diego Padres alternate
    "SFG": "SF",    # San Francisco Giants alternate
    "TBR": "TB",    # Tampa Bay Rays alternate
    "WSN": "WSH",   # Washington Nationals alternate
}


# ─────────────────────────────────────────────────────────────────────────────
# Auth client
# ─────────────────────────────────────────────────────────────────────────────

class KalshiAuthClient:
    """Handles Kalshi REST authentication and JWT token management."""

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._email = email or settings.kalshi_email
        self._password = password or settings.kalshi_password
        self._base_url = (base_url or settings.kalshi_api_base_url).rstrip("/")
        self._token: str | None = None

    def login(self) -> str:
        """Authenticate and return a JWT token.

        Raises:
            httpx.HTTPStatusError: if credentials are rejected.
            ValueError: if email or password are not configured.
        """
        if not self._email or not self._password:
            raise ValueError(
                "Kalshi credentials not configured. "
                "Set KALSHI_EMAIL and KALSHI_PASSWORD in .env"
            )

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/login",
                json={"email": self._email, "password": self._password},
            )
            resp.raise_for_status()
            self._token = resp.json()["token"]
            log.info("kalshi_login_ok", email=self._email)
            return self._token

    @property
    def token(self) -> str | None:
        return self._token

    def refresh(self) -> str:
        """Re-authenticate and return a fresh token."""
        log.info("kalshi_token_refresh")
        return self.login()


# ─────────────────────────────────────────────────────────────────────────────
# Market discovery
# ─────────────────────────────────────────────────────────────────────────────

class KalshiMarketDiscovery:
    """Discovers today's MLB market tickers and maps them to game_pks."""

    def __init__(
        self,
        auth: KalshiAuthClient,
        base_url: str | None = None,
    ) -> None:
        self._auth = auth
        self._base_url = (base_url or settings.kalshi_api_base_url).rstrip("/")

    def find_mlb_markets(self, game_date: date | None = None) -> dict[str, int]:
        """Return {kalshi_ticker: game_pk} for today's MLB games.

        Steps:
        1. Fetch today's MLB schedule via MLBStatsClient.
        2. GET /markets?event_ticker=MLBG&status=open from Kalshi.
        3. Parse each ticker, match teams to schedule, assign game_pk.
        """
        target_date = game_date or date.today()

        # Step 1: get today's schedule
        schedule = self._get_schedule(target_date)
        if not schedule:
            log.info("kalshi_discovery_no_schedule", date=str(target_date))
            return {}

        # Build lookup: (home_abbr, away_abbr) -> game_pk
        schedule_lookup: dict[tuple[str, str], int] = {}
        for game in schedule:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            gpk = game.get("game_pk", 0)
            if home and away and gpk:
                schedule_lookup[(home, away)] = gpk

        # Step 2: fetch Kalshi markets
        markets = self._fetch_kalshi_markets()
        if not markets:
            log.info("kalshi_discovery_no_markets")
            return {}

        # Step 3: match
        result: dict[str, int] = {}
        date_str = target_date.strftime("%Y%m%d")

        for market in markets:
            ticker: str = market.get("ticker", "")
            if not ticker or date_str not in ticker:
                continue

            teams = self._parse_teams_from_ticker(ticker)
            if teams is None:
                continue

            home_mlb, away_mlb = teams

            # Try both orderings (ticker format isn't always consistent)
            gpk = schedule_lookup.get(
                (home_mlb, away_mlb)
            ) or schedule_lookup.get((away_mlb, home_mlb))

            if gpk:
                result[ticker] = gpk
                log.info("kalshi_market_matched", ticker=ticker, game_pk=gpk)
            else:
                log.debug(
                    "kalshi_market_unmatched",
                    ticker=ticker,
                    home=home_mlb,
                    away=away_mlb,
                )

        log.info("kalshi_markets_found", count=len(result), date=str(target_date))
        return result

    def _get_schedule(self, game_date: date) -> list[dict]:
        """Fetch today's schedule from MLB Stats API."""
        try:
            from better.data.ingest.mlb_api import MLBStatsClient

            client = MLBStatsClient()
            try:
                return client.get_schedule(game_date)
            finally:
                client.close()
        except Exception as exc:
            log.warning("kalshi_schedule_fetch_failed", error=str(exc))
            return []

    def _fetch_kalshi_markets(self) -> list[dict]:
        """GET /markets for open MLBG event markets."""
        if not self._auth.token:
            return []

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{self._base_url}/markets",
                    params={"event_ticker": "MLBG", "status": "open"},
                    headers={"Authorization": f"Bearer {self._auth.token}"},
                )
                if resp.status_code == 401:
                    self._auth.refresh()
                    # Retry once
                    resp = client.get(
                        f"{self._base_url}/markets",
                        params={"event_ticker": "MLBG", "status": "open"},
                        headers={"Authorization": f"Bearer {self._auth.token}"},
                    )
                resp.raise_for_status()
                data = resp.json()
                markets: list[dict] = data.get("markets", [])
                log.info("kalshi_markets_fetched", total=len(markets))
                return markets
        except Exception as exc:
            log.warning("kalshi_markets_fetch_failed", error=str(exc))
            return []

    def _parse_teams_from_ticker(self, ticker: str) -> tuple[str, str] | None:
        """Extract (home_mlb, away_mlb) from a Kalshi MLB ticker.

        Expected formats (Kalshi is not always consistent):
            MLBG-BOS-20260301-NYY   → away=BOS, home=NYY
            MLBG-NYY-BOS-20260301   → team1=NYY, team2=BOS
        Returns (home, away) in our 3-letter convention, or None if unparseable.
        """
        parts = ticker.split("-")
        # Remove the MLBG prefix
        if parts and parts[0] == "MLBG":
            parts = parts[1:]

        # Collect non-date, non-numeric parts as team codes
        team_parts: list[str] = []
        for part in parts:
            if part.isdigit() or len(part) == 8:  # 8-digit date like 20260301
                continue
            team_parts.append(part)

        if len(team_parts) < 2:
            return None

        away_raw, home_raw = team_parts[0], team_parts[-1]
        home_mlb = KALSHI_TO_MLB.get(home_raw, home_raw)
        away_mlb = KALSHI_TO_MLB.get(away_raw, away_raw)
        return home_mlb, away_mlb


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket feed
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalshiWebSocketFeed:
    """Async WebSocket feed that streams Kalshi MLB odds into odds_snapshots.

    Follows the same pattern as OddsPoller / LiveGamePoller:
    - ``run()`` is the async entry point (call via asyncio.create_task)
    - ``stop()`` sets ``_running = False`` for graceful shutdown
    """

    auth: KalshiAuthClient
    market_map: dict[str, int]          # {kalshi_ticker: game_pk}
    ws_url: str = field(default_factory=lambda: settings.kalshi_ws_url)
    reconnect_delay: float = 5.0        # seconds before reconnect attempt
    heartbeat_interval: float = 30.0   # seconds between ping messages

    _running: bool = field(default=False, init=False, repr=False)
    _cmd_id: int = field(default=0, init=False, repr=False)

    def stop(self) -> None:
        """Signal the feed to stop after the current message."""
        self._running = False
        log.info("kalshi_ws_stop_requested")

    async def run(self) -> None:
        """Main async loop — connects, subscribes, streams.  Reconnects on error."""
        self._running = True
        log.info("kalshi_ws_starting", markets=len(self.market_map))

        while self._running:
            try:
                await self._connect_and_stream()
            except Exception as exc:
                if self._running:
                    log.warning(
                        "kalshi_ws_disconnected",
                        error=str(exc),
                        retry_in=self.reconnect_delay,
                    )
                    await asyncio.sleep(self.reconnect_delay)

        log.info("kalshi_ws_stopped")

    async def _connect_and_stream(self) -> None:
        """Single WebSocket session: connect → auth → subscribe → stream."""
        async with websockets.connect(self.ws_url) as ws:
            log.info("kalshi_ws_connected", url=self.ws_url)

            # Authenticate
            await self._send(ws, "login", {"token": self.auth.token})

            # Subscribe to all discovered tickers
            tickers = list(self.market_map.keys())
            if tickers:
                await self._send(
                    ws,
                    "subscribe",
                    {"channels": ["ticker"], "market_tickers": tickers},
                )
                log.info("kalshi_ws_subscribed", tickers=tickers)

            # Heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))

            try:
                async for raw in ws:
                    if not self._running:
                        break
                    try:
                        msg = json.loads(raw)
                        await self._handle_message(msg)
                    except json.JSONDecodeError as exc:
                        log.warning("kalshi_ws_bad_json", error=str(exc))
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

    async def _send(self, ws: Any, cmd: str, params: dict) -> None:
        """Send a command message to the WebSocket."""
        self._cmd_id += 1
        payload = json.dumps({"id": self._cmd_id, "cmd": cmd, "params": params})
        await ws.send(payload)

    async def _heartbeat(self, ws: Any) -> None:
        """Send periodic ping to keep the connection alive."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                self._cmd_id += 1
                await ws.send(json.dumps({"id": self._cmd_id, "cmd": "ping"}))
            except Exception as exc:
                log.debug("kalshi_heartbeat_error", error=str(exc))
                break

    async def _handle_message(self, msg: dict) -> None:
        """Dispatch incoming WebSocket messages."""
        msg_type = msg.get("type")

        if msg_type == "ticker":
            await asyncio.get_event_loop().run_in_executor(
                None, self._store_ticker, msg.get("msg", {})
            )
        elif msg_type == "error":
            error_code = msg.get("msg", {}).get("code")
            if error_code in ("auth_failed", "unauthorized"):
                log.warning("kalshi_ws_auth_error", re_authing=True)
                self.auth.refresh()
            else:
                log.warning("kalshi_ws_error_msg", msg=msg)
        elif msg_type in ("subscribed", "ok", "pong"):
            log.debug("kalshi_ws_ack", type=msg_type)

    def _store_ticker(self, ticker_msg: dict) -> None:
        """Convert a Kalshi ticker message and insert into odds_snapshots."""
        market_ticker: str = ticker_msg.get("market_ticker", "")
        game_pk = self.market_map.get(market_ticker)
        if game_pk is None:
            return

        yes_bid = ticker_msg.get("yes_bid")
        yes_ask = ticker_msg.get("yes_ask")
        last_price = ticker_msg.get("last_price")

        # Use mid-price if bid/ask available, else fall back to last price
        if yes_bid is not None and yes_ask is not None:
            mid_cents = (yes_bid + yes_ask) / 2.0
        elif last_price is not None:
            mid_cents = float(last_price)
        else:
            return  # No usable price

        home_fair = round(mid_cents / 100.0, 4)
        away_fair = round(1.0 - home_fair, 4)

        # Kalshi doesn't have traditional overround — fee is on settlement
        home_implied = home_fair
        away_implied = away_fair
        overround = 0.0

        # Convert to American odds for schema compatibility
        home_american = implied_prob_to_american(home_fair)
        away_american = implied_prob_to_american(away_fair)

        now = datetime.now(timezone.utc)

        try:
            conn = get_connection()
            next_id = conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM odds_snapshots"
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO odds_snapshots (
                    id, game_pk, captured_at, bookmaker, market,
                    home_odds_american, away_odds_american,
                    home_implied_prob, away_implied_prob,
                    home_fair_prob, away_fair_prob, overround
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    next_id,
                    game_pk,
                    now,
                    "kalshi",
                    "h2h",
                    home_american,
                    away_american,
                    home_implied,
                    away_implied,
                    home_fair,
                    away_fair,
                    overround,
                ],
            )
            log.info(
                "kalshi_ticker_stored",
                game_pk=game_pk,
                ticker=market_ticker,
                home_fair_prob=home_fair,
                mid_cents=mid_cents,
            )
        except Exception as exc:
            log.error("kalshi_store_failed", error=str(exc), ticker=market_ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

async def start_kalshi_feed() -> KalshiWebSocketFeed | None:
    """Authenticate, discover today's MLB markets, and return a ready feed.

    Returns None if Kalshi credentials are not configured in settings.
    The caller is responsible for starting the feed::

        feed = await start_kalshi_feed()
        if feed:
            asyncio.create_task(feed.run())
    """
    if not settings.kalshi_email or not settings.kalshi_password:
        log.info("kalshi_feed_skipped", reason="no credentials configured")
        return None

    try:
        auth = KalshiAuthClient()
        auth.login()

        discovery = KalshiMarketDiscovery(auth)
        market_map = discovery.find_mlb_markets()

        if not market_map:
            log.info("kalshi_feed_skipped", reason="no MLB markets found for today")
            # Return a feed anyway — it will reconnect and re-subscribe as
            # markets open throughout the day.
            return KalshiWebSocketFeed(auth=auth, market_map={})

        return KalshiWebSocketFeed(auth=auth, market_map=market_map)

    except Exception as exc:
        log.error("kalshi_feed_start_failed", error=str(exc))
        return None
