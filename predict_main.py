
import os
import difflib
from predict_2026_Engine import (
    load_teams, load_matches, load_match_team_stats,
    build_team_stats, build_team_match_records, add_strength_ratings, predict_match,
    write_prediction_report, monte_carlo_predict, write_simulation_report, normalize_name,
    refresh_data_files,
)

WELCOME = """
==============================================
  2026 FIFA WORLD CUP SCORE PREDICTOR
==============================================
This predicts a hypothetical matchup between any two teams using
ONLY real 2026 World Cup data (no history from past tournaments).
It looks at each team's goals, expected goals (xG), shots on target,
and possession across every match they've played this tournament,
then estimates a final score and win probability.

How to enter team names:
  - Type the team's full name as it appears in the tournament,
    e.g. Brazil, Argentina, France, Morocco, United States
  - Capitalization doesn't matter (brazil works fine).
  - Not sure of the exact spelling? Press Enter with nothing typed
    to see the full list of valid teams.
==============================================
"""


REQUIRED_FILES = ("teams.csv", "matches.csv", "match_team_stats.csv")


def _find_data_folder() -> str:
    # look for the CSVs next to this script, or in a data2026/ subfolder.
    # if neither has them yet, make data2026/ so refresh_data_files() has
    # somewhere to download into
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(script_dir, "data2026"), script_dir]
    for folder in candidates:
        if all(os.path.exists(os.path.join(folder, f)) for f in REQUIRED_FILES):
            return folder

    default_folder = os.path.join(script_dir, "data2026")
    os.makedirs(default_folder, exist_ok=True)
    return default_folder


def load_all_stats(folder: str):
    teams = load_teams(os.path.join(folder, "teams.csv"))
    matches = load_matches(os.path.join(folder, "matches.csv"))
    match_team_stats = load_match_team_stats(os.path.join(folder, "match_team_stats.csv"))
    stats = add_strength_ratings(build_team_stats(matches, teams, match_team_stats))
    match_records = build_team_match_records(matches, teams, match_team_stats)
    return stats, match_records


def prompt_team(stats: dict, label: str) -> str:
    valid_names = sorted(stats.keys())
    while True:
        raw = input(f"Enter the {label} team: ").strip()
        if raw == "":
            print("\nValid teams:")
            print(", ".join(valid_names))
            print()
            continue
        name = normalize_name(raw)
        if name in stats:
            return name
        close = difflib.get_close_matches(name, valid_names, n=3, cutoff=0.6)
        if close:
            print(f"Couldn't find '{raw}'. Did you mean: {', '.join(close)}?\n")
        else:
            print(f"Couldn't find '{raw}'. Press Enter with nothing typed to see the full team list.\n")


def main():
    print(WELCOME)
    folder = _find_data_folder()

    print("Checking for the latest completed matches...")
    if refresh_data_files(folder):
        print("Data refreshed with the latest results.\n")
    else:
        have_cached_data = all(
            os.path.exists(os.path.join(folder, f)) for f in REQUIRED_FILES
        )
        if have_cached_data:
            print("Couldn't reach the live data source (check your internet connection) "
                  "-- using the data already saved on your computer.\n")
        else:
            print("Couldn't reach the live data source and no local data was found.\n"
                  "Check your internet connection and try again, or manually place "
                  "teams.csv, matches.csv, and match_team_stats.csv in a 'data2026' "
                  "folder next to this script.\n")
            return

    stats, match_records = load_all_stats(folder)

    while True:
        team_a = prompt_team(stats, "first")
        team_b = prompt_team(stats, "second")
        if team_a == team_b:
            print("Please enter two different teams.\n")
            continue

        result = predict_match(team_a, team_b, stats)
        print()
        print(write_prediction_report(result))
        print()

        print("Running 10,000 simulated matches...")
        sim = monte_carlo_predict(team_a, team_b, stats, match_records, n_simulations=10000)
        print()
        print(write_simulation_report(sim))
        print()

        again = input("Predict another matchup? (y/n): ").strip().lower()
        if again != "y":
            print("Thanks for using the 2026 World Cup Predictor!")
            break
        print()


if __name__ == "__main__":
    main()
