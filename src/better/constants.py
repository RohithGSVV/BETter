"""MLB constants: team codes, park IDs, position maps."""

# Active MLB team abbreviations (as of 2024)
TEAM_ABBREVS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
    "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
    "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SFG",
    "SEA", "STL", "TBR", "TEX", "TOR", "WSN",
]

# Retrosheet team codes → standard abbreviations
RETROSHEET_TEAM_MAP = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCA": "KCR",
    "ANA": "LAA", "LAN": "LAD", "MIA": "MIA", "FLO": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYA": "NYY", "NYN": "NYM",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDN": "SDP",
    "SFN": "SFG", "SEA": "SEA", "SLN": "STL", "TBA": "TBR",
    "TEX": "TEX", "TOR": "TOR", "WAS": "WSN", "MON": "WSN",
}

# MLB Stats API team IDs
MLB_API_TEAM_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KCR": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SFG": 137, "SEA": 136,
    "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120,
}

# Park IDs with known significant park factors
PARK_FACTORS = {
    "COL": {"runs": 1.15, "hr": 1.17},  # Coors Field
    "CIN": {"runs": 1.07, "hr": 1.12},  # GABP
    "TEX": {"runs": 1.05, "hr": 1.08},  # Globe Life
    "MIA": {"runs": 0.92, "hr": 0.88},  # loanDepot park
    "SFG": {"runs": 0.93, "hr": 0.85},  # Oracle Park
    "OAK": {"runs": 0.94, "hr": 0.90},  # Oakland Coliseum
}

# Handedness codes
HAND_LEFT = "L"
HAND_RIGHT = "R"
HAND_SWITCH = "S"

# At-bat outcome categories for Monte Carlo simulation
AT_BAT_OUTCOMES = [
    "strikeout", "walk", "single", "double", "triple", "home_run",
    "groundout", "flyout", "lineout", "pop_out", "hit_by_pitch",
    "sacrifice_fly", "sacrifice_bunt", "error", "fielders_choice",
    "double_play",
]

# Base-out states for win expectancy (runners encoded as 3-bit: 1B, 2B, 3B)
RUNNER_STATES = {
    "---": 0b000,
    "1--": 0b100,
    "-2-": 0b010,
    "--3": 0b001,
    "12-": 0b110,
    "1-3": 0b101,
    "-23": 0b011,
    "123": 0b111,
}

# Season structure
GAMES_PER_SEASON = 162
INNINGS_PER_GAME = 9
OUTS_PER_INNING = 3

# Home field advantage (logit scale ≈ 54% implied)
HOME_FIELD_ADVANTAGE_LOGIT = 0.16
HOME_FIELD_ADVANTAGE_PCT = 0.54

# Elo system constants (FiveThirtyEight-style)
ELO_K_FACTOR = 4
ELO_HOME_ADVANTAGE = 24
ELO_MEAN = 1500
ELO_REVERSION_FACTOR = 1 / 3  # Regress 1/3 to mean between seasons
