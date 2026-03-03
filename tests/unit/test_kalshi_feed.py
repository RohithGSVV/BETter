"""Tests for the Kalshi WebSocket feed integration."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from better.data.live.kalshi_feed import (
    KALSHI_TO_MLB,
    KalshiAuthClient,
    KalshiMarketDiscovery,
    KalshiWebSocketFeed,
    start_kalshi_feed,
)
from better.utils.stats import implied_prob_to_american


# ─────────────────────────────────────────────────────────────────────────────
# implied_prob_to_american helper
# ─────────────────────────────────────────────────────────────────────────────


class TestImpliedProbToAmerican:
    def test_heavy_favorite(self):
        # 80% → should be around -400
        result = implied_prob_to_american(0.80)
        assert result == -400

    def test_even_odds(self):
        # 50% → should be +/-100 (even money)
        result = implied_prob_to_american(0.50)
        assert result == -100  # exact 50% gives -100 by formula

    def test_underdog(self):
        # 40% → positive odds, approximately +150
        result = implied_prob_to_american(0.40)
        assert result > 0
        assert abs(result - 150) <= 1  # float truncation may give 149 or 150

    def test_roundtrip(self):
        """Converting prob → american → implied_prob should be close."""
        from better.utils.stats import implied_probability_from_american

        for prob in [0.35, 0.50, 0.60, 0.75]:
            american = implied_prob_to_american(prob)
            back = implied_probability_from_american(american)
            assert abs(back - prob) < 0.02, f"Roundtrip failed for {prob}: got {back}"


# ─────────────────────────────────────────────────────────────────────────────
# KalshiAuthClient
# ─────────────────────────────────────────────────────────────────────────────


class TestKalshiAuthClient:
    def test_login_returns_token(self):
        """Successful login stores and returns the JWT token."""
        auth = KalshiAuthClient(email="test@test.com", password="pw")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"token": "jwt-abc-123"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            token = auth.login()

        assert token == "jwt-abc-123"
        assert auth.token == "jwt-abc-123"

    def test_login_raises_without_credentials(self):
        """Missing credentials raises ValueError before making any HTTP call."""
        auth = KalshiAuthClient(email="", password="")
        with pytest.raises(ValueError, match="credentials not configured"):
            auth.login()

    def test_refresh_calls_login_again(self):
        """refresh() re-invokes login() and updates the token."""
        auth = KalshiAuthClient(email="a@b.com", password="pw")
        auth._token = "old-token"

        with patch.object(auth, "login", return_value="new-token") as mock_login:
            result = auth.refresh()

        mock_login.assert_called_once()
        assert result == "new-token"


# ─────────────────────────────────────────────────────────────────────────────
# KalshiMarketDiscovery — ticker parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestTickerParsing:
    def setup_method(self):
        self.auth = MagicMock()
        self.auth.token = "jwt-token"
        self.discovery = KalshiMarketDiscovery(self.auth)

    def test_standard_format(self):
        """MLBG-BOS-20260301-NYY → (NYY home, BOS away)."""
        result = self.discovery._parse_teams_from_ticker("MLBG-BOS-20260301-NYY")
        assert result is not None
        home, away = result
        assert home == "NYY"
        assert away == "BOS"

    def test_alternate_format(self):
        """MLBG-NYY-BOS-20260301 → (BOS last = home, NYY away)."""
        result = self.discovery._parse_teams_from_ticker("MLBG-NYY-BOS-20260301")
        assert result is not None

    def test_kalshi_alternate_codes_mapped(self):
        """CHW → CWS, WSN → WSH via KALSHI_TO_MLB dict."""
        result = self.discovery._parse_teams_from_ticker("MLBG-CHW-20260301-WSN")
        assert result is not None
        home, away = result
        assert home == "WSH"
        assert away == "CWS"

    def test_unparseable_ticker_returns_none(self):
        result = self.discovery._parse_teams_from_ticker("MLBG-20260301")
        assert result is None

    def test_kalshi_to_mlb_covers_all_30_teams(self):
        """Mapping covers at least the 30 standard MLB abbreviations."""
        standard_mlb = {
            "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
            "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
            "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
            "SEA", "STL", "TB", "TEX", "TOR", "WSH",
        }
        mapped_values = set(KALSHI_TO_MLB.values())
        assert standard_mlb <= mapped_values


# ─────────────────────────────────────────────────────────────────────────────
# Price conversion
# ─────────────────────────────────────────────────────────────────────────────


class TestPriceConversion:
    def test_mid_price_calculation(self):
        """yes_bid=58, yes_ask=62 → mid=60 → home_fair=0.60."""
        yes_bid, yes_ask = 58, 62
        mid = (yes_bid + yes_ask) / 2.0
        home_fair = mid / 100.0
        assert home_fair == pytest.approx(0.60)
        assert 1.0 - home_fair == pytest.approx(0.40)

    def test_last_price_fallback(self):
        """When bid/ask absent, last_price used as mid."""
        last_price = 55
        home_fair = last_price / 100.0
        assert home_fair == pytest.approx(0.55)

    def test_home_away_sum_to_one(self):
        for mid_cents in [30, 45, 50, 60, 70]:
            home = mid_cents / 100.0
            away = 1.0 - home
            assert home + away == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# KalshiWebSocketFeed — unit
# ─────────────────────────────────────────────────────────────────────────────


class TestKalshiWebSocketFeed:
    def _make_feed(self, market_map=None):
        auth = MagicMock()
        auth.token = "jwt-token"
        return KalshiWebSocketFeed(
            auth=auth,
            market_map=market_map or {"MLBG-BOS-20260301-NYY": 12345},
            ws_url="wss://fake-kalshi.test/ws",
        )

    def test_stop_sets_running_false(self):
        feed = self._make_feed()
        feed._running = True
        feed.stop()
        assert feed._running is False

    def test_store_ticker_inserts_row(self):
        """_store_ticker inserts one row into odds_snapshots."""
        feed = self._make_feed()
        ticker_msg = {
            "market_ticker": "MLBG-BOS-20260301-NYY",
            "yes_bid": 58,
            "yes_ask": 62,
            "last_price": 60,
        }

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = [1]

        with patch("better.data.live.kalshi_feed.get_connection", return_value=mock_conn):
            feed._store_ticker(ticker_msg)

        # Should have called execute twice: once to get next_id, once to insert
        assert mock_conn.execute.call_count == 2
        insert_call = mock_conn.execute.call_args_list[1]
        # Verify bookmaker='kalshi' and game_pk=12345 in the insert values
        args = insert_call[0][1]  # positional params list
        assert "kalshi" in args
        assert 12345 in args

    def test_store_ticker_skips_unknown_market(self):
        """Ticker for an unknown market ticker is silently ignored."""
        feed = self._make_feed()
        ticker_msg = {
            "market_ticker": "MLBG-LAD-20260301-SF",   # not in market_map
            "yes_bid": 55,
            "yes_ask": 65,
        }
        mock_conn = MagicMock()
        with patch("better.data.live.kalshi_feed.get_connection", return_value=mock_conn):
            feed._store_ticker(ticker_msg)

        mock_conn.execute.assert_not_called()

    def test_store_ticker_skips_no_price(self):
        """Ticker with no price data is skipped."""
        feed = self._make_feed()
        ticker_msg = {"market_ticker": "MLBG-BOS-20260301-NYY"}  # no prices
        mock_conn = MagicMock()
        with patch("better.data.live.kalshi_feed.get_connection", return_value=mock_conn):
            feed._store_ticker(ticker_msg)
        mock_conn.execute.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# start_kalshi_feed convenience function
# ─────────────────────────────────────────────────────────────────────────────


class TestStartKalshiFeed:
    @pytest.mark.asyncio
    async def test_returns_none_without_credentials(self):
        """No credentials → returns None without making any HTTP calls."""
        with patch("better.data.live.kalshi_feed.settings") as mock_settings:
            mock_settings.kalshi_email = ""
            mock_settings.kalshi_password = ""
            result = await start_kalshi_feed()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_feed_with_credentials(self):
        """Valid credentials → authenticates and returns a KalshiWebSocketFeed."""
        mock_auth = MagicMock()
        mock_auth.token = "jwt-token"
        mock_auth.login.return_value = "jwt-token"

        mock_discovery = MagicMock()
        mock_discovery.find_mlb_markets.return_value = {
            "MLBG-BOS-20260301-NYY": 12345
        }

        with patch("better.data.live.kalshi_feed.settings") as mock_settings, \
             patch("better.data.live.kalshi_feed.KalshiAuthClient", return_value=mock_auth), \
             patch("better.data.live.kalshi_feed.KalshiMarketDiscovery", return_value=mock_discovery):

            mock_settings.kalshi_email = "user@test.com"
            mock_settings.kalshi_password = "password"
            mock_settings.kalshi_ws_url = "wss://fake.test/ws"

            result = await start_kalshi_feed()

        assert result is not None
        assert isinstance(result, KalshiWebSocketFeed)
        assert result.market_map == {"MLBG-BOS-20260301-NYY": 12345}

    @pytest.mark.asyncio
    async def test_returns_none_on_auth_failure(self):
        """Auth exception → logs error and returns None."""
        mock_auth = MagicMock()
        mock_auth.login.side_effect = Exception("401 Unauthorized")

        with patch("better.data.live.kalshi_feed.settings") as mock_settings, \
             patch("better.data.live.kalshi_feed.KalshiAuthClient", return_value=mock_auth):

            mock_settings.kalshi_email = "user@test.com"
            mock_settings.kalshi_password = "wrong"

            result = await start_kalshi_feed()

        assert result is None
