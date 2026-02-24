"""DuckDB schema definitions and migration."""

from better.data.db import execute


def create_all_tables() -> None:
    """Create all database tables if they don't exist."""
    _create_games_table()
    _create_player_ids_table()
    _create_statcast_table()
    _create_team_batting_table()
    _create_team_pitching_table()
    _create_player_batting_table()
    _create_player_pitching_table()
    _create_team_features_daily_table()
    _create_pitcher_features_table()
    _create_predictions_table()
    _create_odds_snapshots_table()
    _create_win_expectancy_table()
    _create_bet_log_table()


def _create_games_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_pk BIGINT PRIMARY KEY,
            game_date DATE NOT NULL,
            season INTEGER NOT NULL,
            home_team VARCHAR(3) NOT NULL,
            away_team VARCHAR(3) NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            home_win BOOLEAN,
            home_sp_id INTEGER,
            away_sp_id INTEGER,
            home_sp_name VARCHAR(100),
            away_sp_name VARCHAR(100),
            park_id VARCHAR(10),
            attendance INTEGER,
            game_duration_minutes INTEGER,
            day_night VARCHAR(1),
            home_runs_scored INTEGER,
            home_runs_allowed INTEGER,
            away_runs_scored INTEGER,
            away_runs_allowed INTEGER,
            home_hits INTEGER,
            away_hits INTEGER,
            home_errors INTEGER,
            away_errors INTEGER,
            home_walks INTEGER,
            away_walks INTEGER,
            home_strikeouts INTEGER,
            away_strikeouts INTEGER,
            home_home_runs INTEGER,
            away_home_runs INTEGER,
            innings_played INTEGER DEFAULT 9,
            is_postseason BOOLEAN DEFAULT FALSE,
            data_source VARCHAR(20) DEFAULT 'retrosheet',
            home_sp_retrosheet_id VARCHAR(10),
            away_sp_retrosheet_id VARCHAR(10)
        )
    """)


def _create_player_ids_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS player_ids (
            player_id INTEGER PRIMARY KEY,
            mlb_id INTEGER,
            retrosheet_id VARCHAR(10),
            fangraphs_id INTEGER,
            bbref_id VARCHAR(15),
            name_first VARCHAR(50),
            name_last VARCHAR(50),
            birth_date DATE,
            throws VARCHAR(1),
            bats VARCHAR(1)
        )
    """)


def _create_statcast_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS statcast_pitches (
            game_pk INTEGER,
            at_bat_number INTEGER,
            pitch_number INTEGER,
            game_date DATE,
            pitcher_id INTEGER,
            batter_id INTEGER,
            pitcher_name VARCHAR(100),
            batter_name VARCHAR(100),
            pitch_type VARCHAR(5),
            release_speed FLOAT,
            release_spin_rate FLOAT,
            plate_x FLOAT,
            plate_z FLOAT,
            launch_speed FLOAT,
            launch_angle FLOAT,
            hit_distance FLOAT,
            events VARCHAR(30),
            description VARCHAR(50),
            zone INTEGER,
            stand VARCHAR(1),
            p_throws VARCHAR(1),
            home_team VARCHAR(3),
            away_team VARCHAR(3),
            inning INTEGER,
            inning_topbot VARCHAR(3),
            outs_when_up INTEGER,
            balls INTEGER,
            strikes INTEGER,
            on_1b INTEGER,
            on_2b INTEGER,
            on_3b INTEGER,
            estimated_ba_using_speedangle FLOAT,
            estimated_woba_using_speedangle FLOAT,
            woba_value FLOAT,
            woba_denom FLOAT,
            delta_run_exp FLOAT,
            PRIMARY KEY (game_pk, at_bat_number, pitch_number)
        )
    """)


def _create_team_batting_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS team_batting (
            team VARCHAR(3),
            season INTEGER,
            games INTEGER,
            plate_appearances INTEGER,
            at_bats INTEGER,
            hits INTEGER,
            doubles INTEGER,
            triples INTEGER,
            home_runs INTEGER,
            runs INTEGER,
            rbi INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            stolen_bases INTEGER,
            batting_avg FLOAT,
            obp FLOAT,
            slg FLOAT,
            ops FLOAT,
            woba FLOAT,
            wrc_plus FLOAT,
            iso FLOAT,
            babip FLOAT,
            PRIMARY KEY (team, season)
        )
    """)


def _create_team_pitching_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS team_pitching (
            team VARCHAR(3),
            season INTEGER,
            games INTEGER,
            innings_pitched FLOAT,
            era FLOAT,
            fip FLOAT,
            xfip FLOAT,
            siera FLOAT,
            whip FLOAT,
            k_per_9 FLOAT,
            bb_per_9 FLOAT,
            hr_per_9 FLOAT,
            k_pct FLOAT,
            bb_pct FLOAT,
            k_minus_bb_pct FLOAT,
            avg_against FLOAT,
            runs_allowed INTEGER,
            earned_runs INTEGER,
            PRIMARY KEY (team, season)
        )
    """)


def _create_player_batting_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS player_batting (
            player_id INTEGER,
            season INTEGER,
            team VARCHAR(3),
            games INTEGER,
            plate_appearances INTEGER,
            at_bats INTEGER,
            hits INTEGER,
            doubles INTEGER,
            triples INTEGER,
            home_runs INTEGER,
            runs INTEGER,
            rbi INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            batting_avg FLOAT,
            obp FLOAT,
            slg FLOAT,
            ops FLOAT,
            woba FLOAT,
            wrc_plus FLOAT,
            iso FLOAT,
            babip FLOAT,
            war FLOAT,
            bats VARCHAR(1),
            PRIMARY KEY (player_id, season, team)
        )
    """)


def _create_player_pitching_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS player_pitching (
            player_id INTEGER,
            season INTEGER,
            team VARCHAR(3),
            games INTEGER,
            games_started INTEGER,
            innings_pitched FLOAT,
            wins INTEGER,
            losses INTEGER,
            era FLOAT,
            fip FLOAT,
            xfip FLOAT,
            siera FLOAT,
            whip FLOAT,
            k_per_9 FLOAT,
            bb_per_9 FLOAT,
            k_pct FLOAT,
            bb_pct FLOAT,
            k_minus_bb_pct FLOAT,
            war FLOAT,
            stuff_plus FLOAT,
            throws VARCHAR(1),
            is_starter BOOLEAN,
            PRIMARY KEY (player_id, season, team)
        )
    """)


def _create_team_features_daily_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS team_features_daily (
            team VARCHAR(3),
            as_of_date DATE,
            bayesian_strength FLOAT,
            bayesian_strength_var FLOAT,
            pythag_win_pct_30 FLOAT,
            run_diff_30 FLOAT,
            run_diff_14 FLOAT,
            run_diff_7 FLOAT,
            ewma_win_rate_7 FLOAT,
            ewma_win_rate_14 FLOAT,
            ewma_win_rate_30 FLOAT,
            obp_rolling_30 FLOAT,
            wrc_plus_rolling_30 FLOAT,
            team_fip_rolling_30 FLOAT,
            bullpen_fip_composite FLOAT,
            bullpen_ip_last_3d FLOAT,
            games_played_season INTEGER,
            wins_season INTEGER,
            losses_season INTEGER,
            PRIMARY KEY (team, as_of_date)
        )
    """)


def _create_pitcher_features_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS pitcher_features (
            pitcher_id INTEGER,
            as_of_date DATE,
            siera FLOAT,
            fip FLOAT,
            xfip FLOAT,
            stuff_plus FLOAT,
            k_pct FLOAT,
            bb_pct FLOAT,
            k_minus_bb_pct FLOAT,
            ip_last_30d FLOAT,
            days_rest INTEGER,
            game_score_avg_5 FLOAT,
            throws VARCHAR(1),
            PRIMARY KEY (pitcher_id, as_of_date)
        )
    """)


def _create_predictions_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            game_pk BIGINT PRIMARY KEY,
            game_date DATE,
            home_team VARCHAR(3),
            away_team VARCHAR(3),
            bayesian_prob FLOAT,
            gbm_prob FLOAT,
            monte_carlo_prob FLOAT,
            meta_prob FLOAT,
            market_implied_prob FLOAT,
            edge FLOAT,
            kelly_bet_size FLOAT,
            prediction_timestamp TIMESTAMP,
            model_version VARCHAR(20),
            actual_result BOOLEAN,
            prediction_correct BOOLEAN
        )
    """)


def _create_odds_snapshots_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id BIGINT,
            game_pk INTEGER,
            captured_at TIMESTAMP,
            bookmaker VARCHAR(30),
            market VARCHAR(10),
            home_odds_american INTEGER,
            away_odds_american INTEGER,
            home_implied_prob FLOAT,
            away_implied_prob FLOAT,
            home_fair_prob FLOAT,
            away_fair_prob FLOAT,
            overround FLOAT
        )
    """)
    execute("""
        CREATE SEQUENCE IF NOT EXISTS odds_id_seq START 1
    """)


def _create_win_expectancy_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS win_expectancy (
            inning INTEGER,
            half VARCHAR(3),
            outs INTEGER,
            runners INTEGER,
            score_diff INTEGER,
            win_prob FLOAT,
            sample_size INTEGER,
            PRIMARY KEY (inning, half, outs, runners, score_diff)
        )
    """)


def _create_bet_log_table() -> None:
    execute("""
        CREATE TABLE IF NOT EXISTS bet_log (
            id BIGINT,
            game_pk INTEGER,
            game_date DATE,
            bet_side VARCHAR(4),
            model_prob FLOAT,
            market_odds_american INTEGER,
            implied_prob FLOAT,
            edge FLOAT,
            kelly_fraction FLOAT,
            bet_amount FLOAT,
            bankroll_before FLOAT,
            outcome BOOLEAN,
            pnl FLOAT,
            bankroll_after FLOAT
        )
    """)
    execute("""
        CREATE SEQUENCE IF NOT EXISTS bet_log_id_seq START 1
    """)
