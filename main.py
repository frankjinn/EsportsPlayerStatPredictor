import json
import pickle

from sklearn.linear_model import Lasso

from feature_creation import (filter_losses, filter_wins, generate_target,
                              last_n_stats_per_player)
from fetch_raw_data import get_raw_data_df
from imputer import Imputer
from model_eval import eval_predictions
from stat_aggregator import stat_aggregator

#Pipeline param
STATS = ["shots", "goals", "saves", "assists", "score",
         "shootingpercentage", "totaldistance", 'winresult']
EXTERNAL_STATS = ["Goals", "Saves"]
PREV_STATS = ["shots", "goals", "saves", "assists", "score"]
ODDS_TABLE = "rocketleague-odds"
PREGAME_TABLE = "pregamerocketleague_completed_matches"
SUBSTRINGS = ["shots_last", "goals_last", "saves_last", "assists_last", "score_last",
              "shootingpercentage_last", "totaldistance_last", 'winresult_last']

# Step 1: Fetch and save raw data
df_raw_rl = get_raw_data_df(PREGAME_TABLE, ODDS_TABLE,
                            result_keys=["shots", "goals", "saves", "assists",
                                         "score", "shooting_percentage", "total_distance"],
                            testing=True, odds=False, use_overtime=False)
df_raw_rl.to_csv("rocketleague_raw_data.csv")

# Step 2: Data preprocessing
# Calculate statistics per player, filtering by wins and losses
df_processed_1 = last_n_stats_per_player(df_raw_rl,
                                         STATS,
                                         n_list=[2, 3, 5, 10], use_games=True, custom_suffix=None,
                                         filter_condition=None, groupby=["player_id"])

df_processed_1 = last_n_stats_per_player(df_processed_1,
                                         STATS,
                                         n_list=[2, 3, 5, 10], use_games=True, custom_suffix="win",
                                         filter_condition=filter_wins, groupby=["player_id"])

df_processed_1 = last_n_stats_per_player(df_processed_1,
                                         STATS,
                                         n_list=[2, 3, 5, 10], use_games=True, custom_suffix="loss",
                                         filter_condition=filter_losses, groupby=["player_id"])

df_processed_1 = last_n_stats_per_player(df_processed_1,
                                         STATS,
                                         n_list=[2, 3, 5, 10], use_games=True, custom_suffix="matchup",
                                         groupby=["player_id", "team_id", "opposing_id"])

df_processed_1 = generate_target(df_processed_1,
                                 STATS,
                                 use_median=False)

# Step 3: Data imputation
for k in df_processed_1.columns:
    if 'win_result' in k:
        df_processed_1[k.replace("win_result", "winresult")
                       ] = df_processed_1[k]

for k in df_processed_1.columns:
    if 'shooting_percentage' in k:
        df_processed_1[k.replace(
            "shooting_percentage", "shootingpercentage")] = df_processed_1[k]

for k in df_processed_1.columns:
    if 'total_distance' in k:
        df_processed_1[k.replace(
            "total_distance", "totaldistance")] = df_processed_1[k]

matching_columns = []
# Iterate through the columns and check for substrings
for column in df_processed_1.columns:
    if any(substring in column for substring in SUBSTRINGS) and not "expected" in column:
        matching_columns.append(column)

imputer = Imputer(df_processed_1, matching_columns, use_median=True, role=None,
                  tournament_imputation=True, blacklist_tournaments=[])

df_filled = imputer.fill_missing_values()
df_filled = imputer.fill_centered_std(df_filled)

# Output the imputer statistics as JSON
json_file = imputer.convert_to_json_serializable()

with open("imputer.json", 'w') as file:
    json.dump(json_file, file, indent=4)

# Step 4: Statistical aggregation
df_processed_final = stat_aggregator(df_filled,
                                     STATS,
                                     decaying_n_stat_weight=0.9, decaying_match_stat_weight=1)

# Step 5: Model training

oos = df_processed_final[df_processed_final['game_date'] >= '2024-02-01']
in_sample = df_processed_final[df_processed_final['game_date'] < '2024-02-01']

# Running the pipeline
model = Lasso()
distributions = eval_predictions(model, STATS, oos, in_sample, True, True, EXTERNAL_STATS,
                                 PREV_STATS, ODDS_TABLE, PREGAME_TABLE, testing=True)


with open("stats_distributions_rl.pickle", "wb") as f:
    pickle.dump(distributions, f)
