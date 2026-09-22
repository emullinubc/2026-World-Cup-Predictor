"""
predict_2026_Engine.py

The actual number-crunching behind the predictor. Rates each team's attack
and defense relative to the tournament average (goals, xG, shots on target,
possession) and uses that to run a Poisson model + Monte Carlo simulation.

Stdlib only, no pip installs. Data source: mominullptr/FIFA-World-Cup-2026-Dataset
on GitHub (matches.csv, teams.csv, match_team_stats.csv).
"""

import csv
import math
import os
import random
import urllib.request
from collections import defaultdict, Counter

NAME_ALIASES = {
    "usa": "United States",
    "us": "United States",
    "south korea": "South Korea",
    "korea republic": "South Korea",
}

DATA_SOURCE_URLS = {
    "teams.csv": "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/teams.csv",
    "matches.csv": "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/matches.csv",
    "match_team_stats.csv": "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/match_team_stats.csv",
}


def refresh_data_files(folder: str, timeout: int = 10) -> bool:
    # pulls the latest CSVs into `folder`, overwriting whatever's there.
    # returns False if the source is unreachable so the caller can fall
    # back to whatever's already cached on disk
    try:
        for filename, url in DATA_SOURCE_URLS.items():
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            with open(os.path.join(folder, filename), "wb") as f:
                f.write(data)
        return True
    except Exception:
        return False


def normalize_name(name: str) -> str:
    return NAME_ALIASES.get(name.strip().lower(), name.strip())


def _to_float(row: dict, key: str, default: float = 0.0) -> float:
    # csv values come in as strings (or missing), this just makes them safe to sum
    val = row.get(key, "")
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_teams(path: str) -> dict:
    # {team_id: normalized_team_name}
    teams = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams[row["team_id"]] = normalize_name(row["team_name"])
    return teams


def load_matches(path: str) -> list:
    # only care about matches that have actually been played
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "Completed"]


def load_match_team_stats(path: str) -> dict:
    # {(match_id, team_id): row_dict} -- for possession/shots/corners lookups
    stats = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stats[(row["match_id"], row["team_id"])] = row
    return stats


def build_team_match_records(matches: list, teams: dict, team_match_stats: dict) -> dict:
    # {team_name: [{gf, ga, xgf, xga, sot_for, sot_against, possession}, ...]}
    # kept as one entry per match (not averaged) so monte_carlo_predict can
    # bootstrap-resample real match performances instead of just one avg number.
    # skips matches that don't have full shot/possession data.
    records = defaultdict(list)
    for m in matches:
        match_id = m["match_id"]
        for side, opp_side in [("home", "away"), ("away", "home")]:
            tid = m[f"{side}_team_id"]
            opp_tid = m[f"{opp_side}_team_id"]
            own = team_match_stats.get((match_id, tid))
            opp = team_match_stats.get((match_id, opp_tid))
            if not own or not opp:
                continue
            gf = _to_float(m, f"{side}_score")
            ga = _to_float(m, f"{opp_side}_score")
            team_name = teams.get(tid, tid)
            records[team_name].append({
                "gf": gf,
                "ga": ga,
                "xgf": _to_float(m, f"{side}_xg", default=gf),
                "xga": _to_float(m, f"{opp_side}_xg", default=ga),
                "sot_for": _to_float(own, "shots_on_target"),
                "sot_against": _to_float(opp, "shots_on_target"),
                "possession": _to_float(own, "possession_pct"),
            })
    return dict(records)


def build_team_stats(matches: list, teams: dict, team_match_stats: dict = None) -> dict:
    # {team_name: {stat_name: value, ...}}, aggregated over every completed match
    raw = defaultdict(lambda: {
        "matches": 0, "gf": 0.0, "ga": 0.0, "xgf": 0.0, "xga": 0.0,
        "sot_for": 0.0, "sot_against": 0.0, "possession": 0.0, "corners_for": 0.0,
        "match_stats_count": 0,
    })

    for m in matches:
        match_id = m["match_id"]
        for side, opp_side in [("home", "away"), ("away", "home")]:
            tid = m[f"{side}_team_id"]
            opp_tid = m[f"{opp_side}_team_id"]
            gf = _to_float(m, f"{side}_score")
            ga = _to_float(m, f"{opp_side}_score")
            xgf = _to_float(m, f"{side}_xg", default=gf)
            xga = _to_float(m, f"{opp_side}_xg", default=ga)

            r = raw[tid]
            r["matches"] += 1
            r["gf"] += gf
            r["ga"] += ga
            r["xgf"] += xgf
            r["xga"] += xga

            if team_match_stats is not None:
                own = team_match_stats.get((match_id, tid))
                opp = team_match_stats.get((match_id, opp_tid))
                if own and opp:
                    r["sot_for"] += _to_float(own, "shots_on_target")
                    r["sot_against"] += _to_float(opp, "shots_on_target")
                    r["possession"] += _to_float(own, "possession_pct")
                    r["corners_for"] += _to_float(own, "corners")
                    r["match_stats_count"] += 1

    stats = {}
    for tid, r in raw.items():
        n = r["matches"]
        sn = r["match_stats_count"] or 1  # avoid div-by-zero if per-match stats are missing
        team_name = teams.get(tid, tid)
        stats[team_name] = {
            "team_id": tid,
            "team": team_name,
            "matches_played": n,
            "goals_for_avg": r["gf"] / n,
            "goals_against_avg": r["ga"] / n,
            "xg_for_avg": r["xgf"] / n,
            "xg_against_avg": r["xga"] / n,
            "sot_for_avg": r["sot_for"] / sn,
            "sot_against_avg": r["sot_against"] / sn,
            "possession_avg": r["possession"] / sn,
            "corners_for_avg": r["corners_for"] / sn,
        }
    return stats


def add_strength_ratings(stats: dict) -> dict:
    # attack = 40% goals-for + 30% xG-for + 20% SOT-for + 10% possession (all as
    # ratios to tournament average, 1.0 = average team)
    # defense = 50% goals-against + 30% xG-against + 20% SOT-against
    # using ratios instead of raw numbers is what lets goals/xG/shots/possession,
    # which are all on totally different scales, get combined into one score.
    # mutates stats in place (adds attack_strength, defense_strength, tournament_avg_goals)
    def tourney_avg(col):
        vals = [s[col] for s in stats.values()]
        return sum(vals) / len(vals) if vals else 1.0

    avg_goals_for = tourney_avg("goals_for_avg")
    avg_xg_for = tourney_avg("xg_for_avg")
    avg_sot_for = tourney_avg("sot_for_avg")
    avg_possession = tourney_avg("possession_avg")
    avg_goals_against = tourney_avg("goals_against_avg")
    avg_xg_against = tourney_avg("xg_against_avg")
    avg_sot_against = tourney_avg("sot_against_avg")

    for s in stats.values():
        s["attack_strength"] = (
            0.4 * (s["goals_for_avg"] / avg_goals_for if avg_goals_for else 1) +
            0.3 * (s["xg_for_avg"] / avg_xg_for if avg_xg_for else 1) +
            0.2 * (s["sot_for_avg"] / avg_sot_for if avg_sot_for else 1) +
            0.1 * (s["possession_avg"] / avg_possession if avg_possession else 1)
        )
        s["defense_strength"] = (
            0.5 * (s["goals_against_avg"] / avg_goals_against if avg_goals_against else 1) +
            0.3 * (s["xg_against_avg"] / avg_xg_against if avg_xg_against else 1) +
            0.2 * (s["sot_against_avg"] / avg_sot_against if avg_sot_against else 1)
        )
        s["tournament_avg_goals"] = avg_goals_for
        # stash the full set of tournament averages too, for reuse by Monte Carlo resampling
        s["_tourney_avgs"] = {
            "goals_for": avg_goals_for, "xg_for": avg_xg_for, "sot_for": avg_sot_for,
            "possession": avg_possession, "goals_against": avg_goals_against,
            "xg_against": avg_xg_against, "sot_against": avg_sot_against,
        }

    return stats


def _poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def predict_match(team_a: str, team_b: str, stats: dict, max_goals: int = 6) -> dict:
    a, b = normalize_name(team_a), normalize_name(team_b)
    row_a, row_b = stats.get(a), stats.get(b)
    if row_a is None or row_b is None:
        missing = a if row_a is None else b
        return {"error": f"'{missing}' not found in 2026 team stats (check spelling / eliminated before playing?)."}

    avg_goals = row_a["tournament_avg_goals"]
    lam_a = avg_goals * row_a["attack_strength"] * row_b["defense_strength"]
    lam_b = avg_goals * row_b["attack_strength"] * row_a["defense_strength"]

    # build the full scoreline grid assuming both teams score independently
    grid = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            grid[(i, j)] = _poisson_pmf(i, lam_a) * _poisson_pmf(j, lam_b)

    most_likely_score = max(grid, key=grid.get)
    p_a_win = sum(p for (i, j), p in grid.items() if i > j)
    p_draw = sum(p for (i, j), p in grid.items() if i == j)
    p_b_win = sum(p for (i, j), p in grid.items() if i < j)

    return {
        "team_a": a, "team_b": b,
        "expected_goals": {a: round(lam_a, 2), b: round(lam_b, 2)},
        "matches_played": {a: row_a["matches_played"], b: row_b["matches_played"]},
        "attack_strength": {a: round(row_a["attack_strength"], 2), b: round(row_b["attack_strength"], 2)},
        "defense_strength": {a: round(row_a["defense_strength"], 2), b: round(row_b["defense_strength"], 2)},
        "most_likely_score": f"{a} {most_likely_score[0]}-{most_likely_score[1]} {b}",
        "win_probability": {a: round(p_a_win * 100, 1), "draw": round(p_draw * 100, 1), b: round(p_b_win * 100, 1)},
    }


def write_prediction_report(pred: dict) -> str:
    if "error" in pred:
        return pred["error"]
    a, b = pred["team_a"], pred["team_b"]
    lines = [
        f"2026 WORLD CUP PREDICTION: {a} vs {b}",
        "=" * 45,
        f"Matches played this tournament -- {a}: {pred['matches_played'][a]}, {b}: {pred['matches_played'][b]}",
        "",
        f"Attack strength (1.0 = tournament average) -- {a}: {pred['attack_strength'][a]}, {b}: {pred['attack_strength'][b]}",
        f"Defense strength (1.0 = average, lower = stingier) -- {a}: {pred['defense_strength'][a]}, {b}: {pred['defense_strength'][b]}",
        "",
        f"Expected goals -- {a}: {pred['expected_goals'][a]}, {b}: {pred['expected_goals'][b]}",
        f"Most likely scoreline: {pred['most_likely_score']}",
        "",
        f"Win probability -- {a}: {pred['win_probability'][a]}%  Draw: {pred['win_probability']['draw']}%  {b}: {pred['win_probability'][b]}%",
    ]
    return "\n".join(lines)


def _poisson_sample(lam: float, rng: random.Random) -> int:
    # Knuth's algorithm - one Poisson-distributed goal count, stdlib random only
    if lam <= 0:
        return 0
    threshold = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= threshold:
            return k - 1


def monte_carlo_predict(team_a: str, team_b: str, stats: dict, match_records: dict,
                         n_simulations: int = 10000, seed: int = None) -> dict:
    # runs n_simulations hypothetical matches, bootstrap-resampling each team's
    # actual matches every time to rebuild its attack/defense strength. that way
    # the sim captures the uncertainty of only having a few matches per team,
    # not just Poisson noise around one fixed number.
    a, b = normalize_name(team_a), normalize_name(team_b)
    if a not in stats or b not in stats:
        missing = a if a not in stats else b
        return {"error": f"'{missing}' not found in 2026 team stats."}
    if a not in match_records or b not in match_records:
        missing = a if a not in match_records else b
        return {"error": f"No per-match shot/possession data available for '{missing}' (needed for simulation)."}

    rng = random.Random(seed)
    avgs = stats[a]["_tourney_avgs"]
    tourney_avg_goals = stats[a]["tournament_avg_goals"]
    records_a, records_b = match_records[a], match_records[b]

    def resampled_strength(records):
        sample = [rng.choice(records) for _ in records]  # bootstrap, same size as real sample
        n = len(sample)
        gf = sum(r["gf"] for r in sample) / n
        ga = sum(r["ga"] for r in sample) / n
        xgf = sum(r["xgf"] for r in sample) / n
        xga = sum(r["xga"] for r in sample) / n
        sot_f = sum(r["sot_for"] for r in sample) / n
        sot_a = sum(r["sot_against"] for r in sample) / n
        poss = sum(r["possession"] for r in sample) / n

        attack = (
            0.4 * (gf / avgs["goals_for"] if avgs["goals_for"] else 1) +
            0.3 * (xgf / avgs["xg_for"] if avgs["xg_for"] else 1) +
            0.2 * (sot_f / avgs["sot_for"] if avgs["sot_for"] else 1) +
            0.1 * (poss / avgs["possession"] if avgs["possession"] else 1)
        )
        defense = (
            0.5 * (ga / avgs["goals_against"] if avgs["goals_against"] else 1) +
            0.3 * (xga / avgs["xg_against"] if avgs["xg_against"] else 1) +
            0.2 * (sot_a / avgs["sot_against"] if avgs["sot_against"] else 1)
        )
        return attack, defense

    a_wins = b_wins = draws = 0
    scoreline_counts = Counter()
    lam_a_samples, lam_b_samples = [], []

    for _ in range(n_simulations):
        attack_a, defense_a = resampled_strength(records_a)
        attack_b, defense_b = resampled_strength(records_b)

        lam_a = tourney_avg_goals * attack_a * defense_b
        lam_b = tourney_avg_goals * attack_b * defense_a
        lam_a_samples.append(lam_a)
        lam_b_samples.append(lam_b)

        goals_a = _poisson_sample(lam_a, rng)
        goals_b = _poisson_sample(lam_b, rng)
        scoreline_counts[(goals_a, goals_b)] += 1

        if goals_a > goals_b:
            a_wins += 1
        elif goals_a < goals_b:
            b_wins += 1
        else:
            draws += 1

    def mean(vals):
        return sum(vals) / len(vals)

    def stdev(vals):
        m = mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    top_scorelines = scoreline_counts.most_common(5)

    return {
        "team_a": a, "team_b": b, "n_simulations": n_simulations,
        "win_probability": {
            a: round(100 * a_wins / n_simulations, 1),
            "draw": round(100 * draws / n_simulations, 1),
            b: round(100 * b_wins / n_simulations, 1),
        },
        "expected_goals": {
            a: {"mean": round(mean(lam_a_samples), 2), "std": round(stdev(lam_a_samples), 2)},
            b: {"mean": round(mean(lam_b_samples), 2), "std": round(stdev(lam_b_samples), 2)},
        },
        "top_scorelines": [
            {"score": f"{a} {i}-{j} {b}", "pct": round(100 * count / n_simulations, 1)}
            for (i, j), count in top_scorelines
        ],
    }


def write_simulation_report(sim: dict) -> str:
    if "error" in sim:
        return sim["error"]
    a, b, n = sim["team_a"], sim["team_b"], sim["n_simulations"]
    eg_a, eg_b = sim["expected_goals"][a], sim["expected_goals"][b]
    lines = [
        f"MONTE CARLO SIMULATION: {a} vs {b} ({n:,} simulated matches)",
        "=" * 45,
        f"Win probability -- {a}: {sim['win_probability'][a]}%  Draw: {sim['win_probability']['draw']}%  {b}: {sim['win_probability'][b]}%",
        "",
        f"Expected goals (mean +/- std across simulations):",
        f"  {a}: {eg_a['mean']} +/- {eg_a['std']}",
        f"  {b}: {eg_b['mean']} +/- {eg_b['std']}",
        "",
        "Most frequent scorelines:",
    ]
    for row in sim["top_scorelines"]:
        lines.append(f"  {row['score']}  --  {row['pct']}% of simulations")
    return "\n".join(lines)
