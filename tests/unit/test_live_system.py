"""Tests for the live prediction system — WE model, manager, live routes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from better.data.live.game_feed import GameState
from better.models.ingame.win_expectancy import (
    LiveWinProbModel,
    _compute_win_prob,
    _partial_half,
    _runner_expected_runs,
)
from better.data.live.manager import LiveGameManager, LiveGameSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Win Expectancy — helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestPartialHalf:
    def test_zero_outs(self):
        assert _partial_half(0) == pytest.approx(1.0)

    def test_one_out(self):
        assert _partial_half(1) == pytest.approx(2 / 3)

    def test_two_outs(self):
        assert _partial_half(2) == pytest.approx(1 / 3)

    def test_three_outs_clamped(self):
        """outs=3 (from MLB API final games) should be clamped to outs=2."""
        assert _partial_half(3) == pytest.approx(1 / 3)

    def test_negative_outs_clamped(self):
        """Negative outs should be clamped to 0."""
        assert _partial_half(-1) == pytest.approx(1.0)


class TestRunnerExpectedRuns:
    def test_bases_empty(self):
        assert _runner_expected_runs(0b000, 0) == 0.0
        assert _runner_expected_runs(0b000, 1) == 0.0
        assert _runner_expected_runs(0b000, 2) == 0.0

    def test_bases_loaded_zero_outs(self):
        assert _runner_expected_runs(0b111, 0) == pytest.approx(1.54)

    def test_runner_on_first_one_out(self):
        assert _runner_expected_runs(0b100, 1) == pytest.approx(0.29)

    def test_runner_on_second_two_outs(self):
        assert _runner_expected_runs(0b010, 2) == pytest.approx(0.19)


class TestComputeWinProb:
    def test_start_of_game_tied(self):
        """Top 1, no outs, no runners, tied — should be near 0.50-0.55 (home advantage)."""
        wp = _compute_win_prob(1, "top", 0, 0, 0, 0.50)
        assert 0.45 < wp < 0.60

    def test_home_big_lead_late(self):
        """Bottom 9, home up 10 — should be near 1.0."""
        wp = _compute_win_prob(9, "bot", 0, 0, 10, 0.50)
        assert wp > 0.95

    def test_away_big_lead_late(self):
        """Top 9, away up 10 — home should be near 0.0."""
        wp = _compute_win_prob(9, "top", 0, 0, -10, 0.50)
        assert wp < 0.05

    def test_tied_bottom_9_runners_on(self):
        """Tied in bottom 9 with runners, home has walk-off advantage."""
        wp = _compute_win_prob(9, "bot", 0, 0b110, 0, 0.50)
        assert wp > 0.5

    def test_clamped_bounds(self):
        """Result should be clamped between 0.001 and 0.999."""
        wp_high = _compute_win_prob(9, "bot", 2, 0, 20, 0.50)
        wp_low = _compute_win_prob(9, "top", 2, 0, -20, 0.50)
        assert 0.001 <= wp_high <= 0.999
        assert 0.001 <= wp_low <= 0.999

    def test_outs_three_does_not_crash(self):
        """outs=3 should not produce NaN or crash — should be clamped."""
        wp = _compute_win_prob(9, "bot", 3, 0, 5, 0.50)
        assert 0.001 <= wp <= 0.999
        assert not np.isnan(wp)

    def test_outs_three_tied_b9_reasonable(self):
        """outs=3 tied B9 should give reasonable prob after clamping (not 25%)."""
        wp = _compute_win_prob(9, "bot", 3, 0, 0, 0.50)
        # After clamping to outs=2, tied B9 should still give home team decent chance
        assert wp > 0.30  # was 0.25 before fix, now clamped to outs=2


# ─────────────────────────────────────────────────────────────────────────────
# LiveWinProbModel
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveWinProbModel:
    def _make_state(self, inning=1, half="top", outs=0, runners=0,
                    home_score=0, away_score=0) -> GameState:
        return GameState(
            game_pk=12345,
            inning=inning, half=half, outs=outs, runners=runners,
            home_score=home_score, away_score=away_score,
        )

    def test_predict_without_loaded_table(self):
        """Without a loaded table, should still return a prediction using fallback."""
        model = LiveWinProbModel()
        state = self._make_state()
        result = model.predict(state)
        assert "win_prob" in result
        assert 0 < result["win_prob"] < 1

    def test_predict_with_pregame_prior(self):
        """Pregame prior should influence early-inning predictions."""
        model = LiveWinProbModel()
        state = self._make_state(inning=1)

        # Strong home favorite (pre-game)
        result_fav = model.predict(state, pregame_prob=0.80)
        # Strong away favorite (pre-game)
        result_dog = model.predict(state, pregame_prob=0.20)

        # In inning 1, pregame gets 60% weight — these should differ
        assert result_fav["win_prob"] > result_dog["win_prob"]

    def test_we_weight_increases_with_inning(self):
        """WE weight should increase as game progresses."""
        model = LiveWinProbModel()
        assert model._get_we_weight(1) == 0.40
        assert model._get_we_weight(5) == 0.70
        assert model._get_we_weight(8) == 0.90
        assert model._get_we_weight(9) == 0.98

    def test_predict_outs_three_fallback(self):
        """outs=3 should still produce valid predictions via clamping."""
        model = LiveWinProbModel()
        state = self._make_state(inning=9, half="bot", outs=3, home_score=5, away_score=3)
        result = model.predict(state)
        assert 0 < result["win_prob"] < 1
        assert not np.isnan(result["win_prob"])

    def test_predict_returns_all_keys(self):
        """Result dict should contain all expected keys."""
        model = LiveWinProbModel()
        state = self._make_state(inning=3, outs=1, home_score=2, away_score=1)
        result = model.predict(state, pregame_prob=0.55)

        expected_keys = {
            "win_prob", "we_prob", "pregame_prob", "we_weight",
            "inning", "half", "outs", "runners",
            "score_diff", "home_score", "away_score",
        }
        assert expected_keys <= set(result.keys())
        assert result["pregame_prob"] == 0.55
        assert result["inning"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# LiveGameSnapshot
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveGameSnapshot:
    def test_to_dict_keys(self):
        snap = LiveGameSnapshot(
            game_pk=12345, home_team="NYY", away_team="BOS",
            timestamp=datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc),
            inning=5, half="top", outs=2, runners=0b100,
            home_score=3, away_score=1,
            win_prob=0.72, we_prob=0.70, pregame_prob=0.55, we_weight=0.70,
            market_prob=0.65, edge=0.07, market_source="kalshi",
        )
        d = snap.to_dict()
        assert d["game_pk"] == 12345
        assert d["home_team"] == "NYY"
        assert d["win_prob"] == 0.72
        assert d["edge"] == 0.07
        assert d["market_source"] == "kalshi"
        assert "timestamp" in d


# ─────────────────────────────────────────────────────────────────────────────
# LiveGameManager
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveGameManager:
    def test_get_snapshot_empty(self):
        manager = LiveGameManager()
        assert manager.get_snapshot(99999) is None

    def test_get_all_snapshots_empty(self):
        manager = LiveGameManager()
        assert manager.get_all_snapshots() == []

    def test_update_market_prob_creates_edge(self):
        """Market prob update should compute edge against model win_prob."""
        manager = LiveGameManager()
        # Manually create a snapshot
        snap = LiveGameSnapshot(
            game_pk=12345, home_team="NYY", away_team="BOS",
            timestamp=datetime.now(timezone.utc),
            inning=5, half="top", outs=1, runners=0,
            home_score=3, away_score=1,
            win_prob=0.72, we_prob=0.70, pregame_prob=0.55, we_weight=0.70,
        )
        manager._snapshots[12345] = snap

        manager.update_market_prob(12345, 0.65, "kalshi")

        updated = manager.get_snapshot(12345)
        assert updated.market_prob == 0.65
        assert updated.edge == pytest.approx(0.07, abs=0.001)
        assert updated.market_source == "kalshi"

    def test_update_market_prob_no_snap(self):
        """Market update for unknown game should store prob without error."""
        manager = LiveGameManager()
        manager.update_market_prob(99999, 0.60)
        # Should not raise, just stores the prob
        assert manager._market_probs[99999] == 0.60

    def test_on_state_change(self):
        """State change should create a new snapshot with predictions."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}
        manager._pregame_probs[12345] = 0.55

        state = GameState(
            game_pk=12345, inning=3, half="top", outs=1, runners=0,
            home_score=1, away_score=0,
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap is not None
        assert snap.home_team == "NYY"
        assert snap.inning == 3
        assert 0 < snap.win_prob < 1
        assert snap.status == "live"

    def test_on_state_change_with_market(self):
        """State change with market data should compute edge."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}
        manager._pregame_probs[12345] = 0.55
        manager._market_probs[12345] = 0.60

        state = GameState(
            game_pk=12345, inning=5, half="bot", outs=0, runners=0,
            home_score=2, away_score=1,
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.edge is not None
        assert snap.market_prob == 0.60

    def test_on_update_callback(self):
        """on_update callback should fire on state change."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}

        callback = MagicMock()
        manager.on_update = callback

        state = GameState(game_pk=12345, inning=1, half="top", outs=0, runners=0,
                          home_score=0, away_score=0)
        manager._on_state_change(12345, state)

        callback.assert_called_once()
        arg = callback.call_args[0][0]
        assert isinstance(arg, LiveGameSnapshot)

    def test_stop(self):
        """stop() should set _running to False."""
        manager = LiveGameManager()
        manager._running = True
        manager.stop()
        assert manager._running is False

    def test_on_state_change_final_home_wins(self):
        """Final game where home wins should have win_prob=1.0 and status='final'."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}
        manager._pregame_probs[12345] = 0.55

        state = GameState(
            game_pk=12345, inning=9, half="bot", outs=2, runners=0,
            home_score=5, away_score=3, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap is not None
        assert snap.status == "final"
        assert snap.win_prob == 1.0
        assert snap.we_prob == 1.0
        assert snap.we_weight == 1.0

    def test_on_state_change_final_away_wins(self):
        """Final game where away wins should have win_prob=0.0."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}

        state = GameState(
            game_pk=12345, inning=9, half="bot", outs=2, runners=0,
            home_score=2, away_score=4, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.status == "final"
        assert snap.win_prob == 0.0

    def test_on_state_change_final_extra_innings(self):
        """Final game in extra innings (12th inning)."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}

        state = GameState(
            game_pk=12345, inning=12, half="bot", outs=2, runners=0,
            home_score=3, away_score=2, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.status == "final"
        assert snap.win_prob == 1.0

    def test_on_state_change_final_walkoff(self):
        """Walk-off win: home wins in bottom of 9th before 3 outs."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}

        state = GameState(
            game_pk=12345, inning=9, half="bot", outs=1, runners=0,
            home_score=4, away_score=3, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.status == "final"
        assert snap.win_prob == 1.0

    def test_on_state_change_final_away_wins_top_extras(self):
        """Away team wins in top of extra inning."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}

        state = GameState(
            game_pk=12345, inning=10, half="top", outs=2, runners=0,
            home_score=2, away_score=5, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.status == "final"
        assert snap.win_prob == 0.0

    def test_on_state_change_final_with_market_edge(self):
        """Final game should still compute edge against market."""
        manager = LiveGameManager()
        manager._game_info[12345] = {"home_team": "NYY", "away_team": "BOS"}
        manager._market_probs[12345] = 0.60

        state = GameState(
            game_pk=12345, inning=9, half="bot", outs=2,
            home_score=5, away_score=3, status="final",
        )
        manager._on_state_change(12345, state)

        snap = manager.get_snapshot(12345)
        assert snap.edge == pytest.approx(0.40, abs=0.001)  # 1.0 - 0.60


# ─────────────────────────────────────────────────────────────────────────────
# GameState
# ─────────────────────────────────────────────────────────────────────────────


class TestGameState:
    def test_score_diff(self):
        state = GameState(game_pk=1, home_score=5, away_score=3)
        assert state.score_diff == 2

    def test_state_key(self):
        state = GameState(game_pk=1, inning=3, half="bot", outs=2, runners=0b110,
                          home_score=4, away_score=2)
        assert state.state_key == (3, "bot", 2, 0b110, 2)

    def test_has_changed(self):
        s1 = GameState(game_pk=1, inning=3, outs=1)
        s2 = GameState(game_pk=1, inning=3, outs=2)
        s3 = GameState(game_pk=1, inning=3, outs=1)
        assert s1.has_changed(s2) is True
        assert s1.has_changed(s3) is False

    def test_default_status_is_live(self):
        state = GameState(game_pk=1)
        assert state.status == "live"

    def test_status_final(self):
        state = GameState(game_pk=1, status="final")
        assert state.status == "final"

    def test_has_changed_detects_status_change(self):
        """Transition from live to final should count as changed."""
        s1 = GameState(game_pk=1, inning=9, outs=2, status="live")
        s2 = GameState(game_pk=1, inning=9, outs=2, status="final")
        assert s1.has_changed(s2) is True

    def test_has_changed_same_status(self):
        s1 = GameState(game_pk=1, inning=5, outs=1, status="live")
        s2 = GameState(game_pk=1, inning=5, outs=1, status="live")
        assert s1.has_changed(s2) is False

    def test_state_key_excludes_status(self):
        """state_key should not include status — it's for WE lookup only."""
        s1 = GameState(game_pk=1, inning=5, outs=1, status="live")
        s2 = GameState(game_pk=1, inning=5, outs=1, status="final")
        assert s1.state_key == s2.state_key


# ─────────────────────────────────────────────────────────────────────────────
# Live API routes
# ─────────────────────────────────────────────────────────────────────────────


class TestLiveRoutes:
    def test_live_games_offline(self):
        """When manager returns None, should return offline status."""
        from fastapi.testclient import TestClient
        from better.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        with patch("better.api.services.get_prediction_service") as mock_svc:
            mock_svc.return_value.get_live_manager.return_value = None
            resp = client.get("/api/live/games")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "offline"
            assert data["games"] == []

    def test_live_games_with_data(self):
        """When manager has snapshots, should return them."""
        from fastapi.testclient import TestClient
        from better.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_snap = MagicMock()
        mock_snap.status = "live"
        mock_snap.to_dict.return_value = {"game_pk": 12345, "status": "live"}

        mock_manager = MagicMock()
        mock_manager.get_all_snapshots.return_value = [mock_snap]

        with patch("better.api.services.get_prediction_service") as mock_svc:
            mock_svc.return_value.get_live_manager.return_value = mock_manager
            resp = client.get("/api/live/games")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "live"
            assert len(data["games"]) == 1
            assert data["active_count"] == 1

    def test_live_game_not_found(self):
        """Request for unknown game_pk should return 404."""
        from fastapi.testclient import TestClient
        from better.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=False)

        mock_manager = MagicMock()
        mock_manager.get_snapshot.return_value = None

        with patch("better.api.services.get_prediction_service") as mock_svc:
            mock_svc.return_value.get_live_manager.return_value = mock_manager
            resp = client.get("/api/live/games/99999")
            assert resp.status_code == 404
