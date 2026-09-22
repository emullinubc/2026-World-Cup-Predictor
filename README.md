## 2026 FIFA World Cup Score Predictor
This project predicts the scoreline and win probabilities for a hypothetical match between any two teams at the 2026 FIFA World Cup. Predictions are built only from data generated during the 2026 tournament, with no results from previous World Cups or international friendlies. Each team's attacking and defensive strength is rated from its goals, expected goals (xG), shots on target, and possession relative to the tournament average, and these ratings are used in a Poisson model of goal scoring.

Since most teams played between 3 and 8 matches, any single strength rating carries considerable uncertainty. To account for this, the project extends the basic Poisson model with a Monte Carlo simulation that bootstrap-resamples each team's actual match performances across 10,000 simulated matches.

## Model
Each team $i$ receives an attack and a defense rating, calculated as a weighted average of its per-match statistics, each divided by the tournament average:

$$Att_i = 0.4\frac{GF_i}{\overline{GF}} + 0.3\frac{xGF_i}{\overline{xGF}} + 0.2\frac{SOT_i}{\overline{SOT}} + 0.1\frac{Poss_i}{\overline{Poss}}$$

$$Def_i = 0.5\frac{GA_i}{\overline{GA}} + 0.3\frac{xGA_i}{\overline{xGA}} + 0.2\frac{SOTA_i}{\overline{SOTA}}$$

Expected goals for team $A$ playing team $B$:

$$\lambda_A = \bar{G} \cdot Att_A \cdot Def_B \qquad \lambda_B = \bar{G} \cdot Att_B \cdot Def_A$$

Each team's goals are then modelled as an independent Poisson variable:

$$P(X_A = k) = \frac{\lambda_A^k e^{-\lambda_A}}{k!}$$

- **$GF_i$, $GA_i$**: goals scored and conceded per match.
- **$xGF_i$, $xGA_i$**: expected goals created and conceded per match.
- **$SOT_i$, $SOTA_i$**: shots on target taken and conceded per match.
- **$Poss_i$**: average possession percentage.
- **$\overline{GF}$, $\overline{xGF}$, etc.**: the average of each statistic across all 48 teams.
- **$\bar{G}$**: the tournament average goals scored per team per match.
- **$\lambda_A$**: team $A$'s expected goals in the matchup.

A rating of 1.0 represents an average team at the tournament. Higher attack ratings are better, while lower defense ratings are better since the defense rating measures what a team concedes.

Each statistic is divided by its tournament average before weighting because goals, shots, and possession are measured on completely different scales. Expressing each as a ratio puts them on a common scale so they can be combined. Actual goals receive the largest weight since they decide matches, but over a handful of games they are noisy. xG and shots on target measure the quality and volume of chances a team creates, which tend to be more stable, and partly separate performance from finishing luck.

Win, draw, and loss probabilities come from summing the probability of every scoreline from 0-0 to 6-6.

For the Monte Carlo simulation, each of the 10,000 iterations resamples each team's matches with replacement, recalculates $Att$ and $Def$ from that resample, and draws a scoreline from the resulting Poisson distributions. The spread of $\lambda$ across iterations shows how sensitive a prediction is to which matches happen to be included.

## Data
Data pulled from the [FIFA World Cup 2026 Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset) (mominullptr), covering all 104 matches played between June 11 and July 19, 2026: `teams.csv` (team names and IDs), `matches.csv` (scores, xG, and match status), and `match_team_stats.csv` (per-team possession, shots, shots on target, and corners). Each time the script runs, it attempts to download the latest version of all three files into `data2026/`, and falls back to the saved copies if the source can't be reached. A copy of the final tournament data is included in this repository.

## Results

**Example: Argentina vs France**

| | Poisson | Monte Carlo |
|---|---|---|
| Argentina win | 45.6% | 44.8% |
| Draw | 24.0% | 23.8% |
| France win | 30.2% | 31.5% |
| Argentina expected goals | 1.66 | 1.67 (± 0.70) |
| France expected goals | 1.31 | 1.31 (± 0.31) |

France has the slightly stronger attack rating (1.70 vs 1.62), but Argentina is favoured because its defense conceded considerably less over the tournament (0.58 vs 0.77). The most likely single scoreline is 1-1 under both methods (10.8% of simulations). This is not a contradiction: probability is spread across many possible scorelines, and more of it falls on Argentina wins combined than on any other outcome.

The two methods agree closely on win probabilities, which is expected since the bootstrap is centred on the same underlying data. The difference is in the spread. Argentina's expected goals vary more than twice as much as France's across simulations, meaning Argentina's attacking output and France's defensive record fluctuated more from match to match. Monte Carlo results will vary slightly between runs.

## Limitations
- Ratings are not adjusted for opponent quality. A team that scored heavily against weak group-stage opponents will appear stronger than it is, while teams that advanced deep into the knockout rounds faced stronger opposition, which may understate their ratings. An iterative rating system that weights performances by opponent strength would be needed to address this.
- The weights (40/30/20/10 for attack, 50/30/20 for defense) are chosen by judgment rather than estimated from data. Different weights would produce different ratings.
- Teams eliminated in the group stage have only three matches of data, so their ratings and bootstrap resamples are based on very little information.
- Goals are assumed to be independent between teams. Basic Poisson models tend to underestimate low-scoring draws, which an adjustment for low-scoring draws would correct for.
- Extra time and penalty shootouts are not simulated, so a draw means level at the end of play.
- Pre-tournament information is excluded by design. This keeps the model focused on tournament form, but discards long-run measures of team quality such as Elo ratings, which are available in the dataset but unused.
- Injuries, suspensions, rest days, and venue are not included.

## Usage
- No dependencies outside the Python standard library (Python 3.8+)
- Run the script: `python3 predict_main.py`
- Enter two teams when prompted. Pressing Enter with nothing typed shows the full list of valid team names
- Running it downloads the latest data into `data2026/`, or uses the saved copy
