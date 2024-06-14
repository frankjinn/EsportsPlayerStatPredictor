import pandas as pd
import json
import numpy as np
import math
import warnings
from dynamodb_json import json_util
warnings.filterwarnings("ignore")
import boto3

dynamodb = boto3.client('dynamodb', aws_access_key_id="", aws_secret_access_key="", region_name="us-east-1")

def get_odds_mapping(list_odds):
    """
    Create a mapping of game IDs to average win probabilities for two teams from a list of odds data.

    Parameters:
    - list_odds (list): A list of odds data containing game information.

    Returns:
    - odds_mapping (dict): A dictionary where game IDs are keys, and values are dictionaries with
      "team_1_win_probs" and "team_2_win_probs" representing the average win probabilities for two teams.
    """
    odds_mapping = {}
    for game in list_odds:
        team_1_win_probs, team_2_win_probs = [], []
        game_id = game["gameid"]["S"]
        for odds_provider in game["odds"]["M"].keys():
            if len(game["odds"]["M"][odds_provider]["L"]) == 0:
                continue
            most_recent_odds = game["odds"]["M"][odds_provider]["L"][-1]["M"]
            team_1_win_probs.append(float(most_recent_odds["team_1_win_prob"]["N"]))
            team_2_win_probs.append(float(most_recent_odds["team_2_win_prob"]["N"]))
        odds_mapping[game_id] = {"team_1_win_probs": np.average(team_1_win_probs), "team_2_win_probs": np.average(team_2_win_probs)}
    return odds_mapping

def get_overtime(item, use_overtime):
    """
    Get a list of overtime flags for a game based on the score data.

    Parameters:
    - item (dict): A dictionary containing game information.
    - use_overtime (bool): Whether to include overtime flags.

    Returns:
    - overtime (list): A list of 0s and 1s indicating overtime for each game round.
    """
    overtime = []
    if not use_overtime:
        return [1 for i in range(len(item["teams"]["L"][0]["M"]["players"]["L"][0]["M"]["results"]["L"]))]
    if "results" not in item["teams"]["L"][0]["M"]:
        return None
    for game in range(len(item["teams"]["L"][0]["M"]["results"]["L"])):
        team_1_score = int(item["teams"]["L"][0]["M"]["results"]["L"][game]["M"]["rounds_won"]["N"])
        team_2_score = int(item["teams"]["L"][1]["M"]["results"]["L"][game]["M"]["rounds_won"]["N"])
        if (team_1_score > 16 or team_2_score > 16) and use_overtime:
            overtime.append(0)
        else:
            overtime.append(1)
    return overtime

def mask_result(data_list, mask_list):
    """
    Mask a list of data based on a list of masks.

    Parameters:
    - data_list (list): A list of data values.
    - mask_list (list): A list of masks.

    Returns:
    - result_list_final (list): A list of data values after applying the mask.
    """
    result_list = [x for x, mask in zip(data_list, mask_list) if mask != 0]
    result_list_final = [x for x in result_list if x != -1000]
    return result_list_final

def calculate_results(game_data, feature_name, overtime):
    """
    Calculate results for a specific game feature based on game data.

    Parameters:
    - game_data (dict): A dictionary containing game data.
    - feature_name (str): The name of the game feature to calculate.
    - overtime (list): A list of overtime flags for game rounds.

    Returns:
    - result (list): A list of calculated results for the feature.
    """
    feature_values = [float(game_data[f][feature_name]["N"]) if feature_name in game_data[f] else -1000 for f in game_data]
    return mask_result(feature_values, overtime)

def process_player_data(player_data, game_id, game_date, win_result, overtime, feature_keys, result_lists):
    """
    Process player data for a game and append results to result lists.

    Parameters:
    - player_data (dict): A dictionary containing player data.
    - game_id (str): The ID of the game.
    - game_date (str): The date of the game.
    - win_result (float): The win result for the player's team.
    - overtime (list): A list of overtime flags for game rounds.
    - feature_keys (list): A list of keys representing game features to process.
    - result_lists (dict): A dictionary of result lists for each game feature.
    """
    results = player_data.get("results", {"L": []})["L"]
    player_results = {key: [] for key in feature_keys}
    for game in results:
        game_data = game["M"]
        if any("NULL" in game_data.get(key, "") for key in feature_keys):
            continue
        for key in feature_keys:
            try: result = float(game_data.get(key, {"N": "-1000"})["N"])
            except KeyError: result = -1000.0
            player_results[key].append(result)
    # Append masked results for all games as arrays
    for key in feature_keys:
        if overtime is None:
            overtime = [1]*len(player_results[key])
        result = mask_result(player_results[key], overtime)
        result_lists[key].append(result)

def process_raw_data(data_frame, odds_mapping, result_keys, use_overtime=True):
    """
    Process raw game data, calculate results, and create a DataFrame.

    Parameters:
    - data_frame (list): A list of dictionaries containing raw game data.
    - odds_mapping (dict): A mapping of game IDs to average win probabilities.
    - result_keys (list): A list of keys representing game features to process.
    - use_overtime (bool): Whether to include overtime flags.

    Returns:
    - df_raw_data (pd.DataFrame): A DataFrame containing processed game data.
    """
    df_raw_data = pd.DataFrame()
    game_ids = []
    game_date = []
    player_id = []
    player_names = []
    team_id = []
    team_name = []
    opposing_id = []
    opposing_name = []
    result_lists = {key: [] for key in result_keys}
    win_result = []
    overtime_list = []
    win_probability = []
    tournament_id = []
    player_roles = []

    for item in data_frame:
        if "teams" in item:
            if not use_overtime: overtime = None
            else: overtime = get_overtime(item, use_overtime)
            for team_idx in range(len(item["teams"]["L"])):
                team = item["teams"]["L"][team_idx]["M"]
                opposing_team = item["teams"]["L"][abs(team_idx - 1)]["M"]
                for player_temp in team["players"]["L"]:
                    player = player_temp["M"]
                    process_player_data(
                        player,
                        item["gameid"]["S"],
                        item["date"]["S"],
                        team.get("win_result", {"N": 0.5}).get("N", 0.5),
                        overtime,
                        result_keys,
                        result_lists
                    )
                    game_ids.append(item["gameid"]["S"])  # Append the game_id here
                    game_date.append(item["date"]["S"])  # Append the game_date here
                    player_id.append(player["id"]["S"])  # Append the player_id here
                    player_names.append(player["username"]["S"])  # Append the player_name here
                    player_roles.append(player.get('role', {}).get('S'))
                    team_id.append(team["id"]["S"])  # Append the team_id here
                    team_name.append(team["name"]["S"])  # Append the team_name here
                    opposing_id.append(opposing_team["id"]["S"])  # Append the opposing_id here
                    opposing_name.append(opposing_team["name"]["S"])  # Append the opposing_name here
                    win_result.append(team.get("win_result", {"N": 0.5}).get("N", 0.5))  # Append the win_result here
                    overtime_list.append(overtime)  # Append the overtime here
                    game_odds = odds_mapping.get(item["gameid"]["S"], {"team_1_win_probs": 0.5, "team_2_win_probs": 0.5}) if odds_mapping is not None else None
                    team_number = str(team_idx+1)
                    win_probability.append(game_odds[f"team_{team_number}_win_probs"] if game_odds is not None else 0.5)
                    if "league_details" in item:
                        tournament_id.append(item["league_details"]["M"]["id"]["S"])
                    else:
                        tournament_id.append(None)
    for key in result_lists.keys():
        print(key, len(result_lists[key]))

    # Create a DataFrame from the collected data
    df_raw_data = pd.DataFrame({
        "game_id": game_ids,
        "game_date": game_date,
        "player_id": player_id,
        "player_name": player_names,
        "team_id": team_id,
        "team_name": team_name,
        "opposing_id": opposing_id,
        "opposing_name": opposing_name,
        "win_result": win_result,
        "win_probability": win_probability,
        "tournament_id": tournament_id,
        "player_role":player_roles,
        **result_lists  # Include the result data in the DataFrame
    })
    return df_raw_data

def get_data_from_table(table_name, testing):
    """
    Get data from a DynamoDB table.

    Parameters:
    - table_name (str): The name of the DynamoDB table to retrieve data from.
    - testing (bool): A flag indicating whether testing mode is enabled.

    Returns:
    - data_list (list): A list of data items retrieved from the table.
    """
    data_list = []
    response = dynamodb.scan(TableName=table_name)
    data_list.extend(response['Items'])
    counter = 0
    if testing == True:
        while 'LastEvaluatedKey' in response and counter < 2:
            response = dynamodb.scan(
                TableName=table_name,
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            data_list.extend(response['Items'])
            counter += 1
        return data_list
    while 'LastEvaluatedKey' in response:
        response = dynamodb.scan(
            TableName=table_name,
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        data_list.extend(response['Items'])
    return data_list

def get_raw_data_df(table_raw_name, table_odds_name, result_keys, filter_out_overtime=True, testing=False, odds=True):
    """
    Get and process raw game data and odds data, and return a DataFrame.

    Parameters:
    - table_raw_name (str): The name of the DynamoDB table containing raw game data.
    - table_odds_name (str): The name of the DynamoDB table containing odds data.
    - result_keys (list): A list of keys representing game features to process.
    - filter_out_overtime (bool): Whether to filter out overtime matches.
    - testing (bool): A flag indicating whether testing mode is enabled.

    Returns:
    - raw_data_df (pd.DataFrame): A DataFrame containing processed game data.
    """
    print("fetching raw data")
    df_raw_data = get_data_from_table(table_raw_name, testing)
    odds_mapping = None
    if odds:
        print("fetching odds data")
        df_odds = get_data_from_table(table_odds_name, testing)
        print("creating odds mapping")
        odds_mapping = get_odds_mapping(df_odds)
    print("creating raw data frame")
    print("df_raw_data:", len(df_raw_data))
    raw_data_df = process_raw_data(df_raw_data, odds_mapping, result_keys, filter_out_overtime)
    print("raw_data_df:", len(raw_data_df))
    raw_data_df['game_date'] = pd.to_datetime(raw_data_df['game_date'])
    raw_data_df = raw_data_df.sort_values(by='game_date', ascending=True).reset_index()
    print("df_raw_data:", len(df_raw_data))
    return raw_data_df

def process_downloaded_files(raw_data_file, odds_file, result_keys = ["kills", "deaths", "assists", "headshots", "KAST", "ADR"], use_overtime=False):
    print("fetching raw data")
    with open(raw_data_file, 'r') as fp:
        df_raw_data = json_util.dumps(json.load(fp))
    print("fetching odds data")
    with open(odds_file, 'r') as fp:
        df_odds = json_util.dumps(json.load(fp))
    odds_mapping = get_odds_mapping(df_odds)
    print("creating raw data frame")
    raw_data_df = process_raw_data(df_raw_data, odds_mapping, result_keys, use_overtime)
    raw_data_df['game_date'] = pd.to_datetime(raw_data_df['game_date'])
    raw_data_df = raw_data_df.sort_values(by='game_date', ascending=True).reset_index()
    return raw_data_df
