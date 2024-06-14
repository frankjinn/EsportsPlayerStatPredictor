"""
This module does pulls in the model, predicts, and evaluates those predictions.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from fetch_raw_data import get_data_from_table
from model import model_pipeline

def get_external_preds(odds_table, stats, median = False):
    """
    Fetch DF with mean/median of other bookmaker's predictions

    Args:
        odds_table (List): Raw data obtained from database.
        stats (List): List of stats that bookmakers have a prediction on.
        median (Bool): Use median of different bookmaker predictions

    Returns:
        df (pd.DataFrame): Dataframe with bookmakers stat predictions indexed on gameid and playerid
    """
    print("Getting external predictions")
    agg = np.median if median else np.mean
    game_ids = []
    player_id = []
    player_names = []
    player_prediction = defaultdict(lambda: [])
    game_date = []
    for item in odds_table:
        if 'teams' in item:
            for team in item['teams']['L']:
                for player in team['M']['players']['L']:
                    if 'projections' in player['M']:
                        game_ids.append(item['gameid']['S'])
                        game_date.append(item['date']['S'])
                        player_id.append(player['M']['id']['S'])
                        player_names.append(player['M']['username']['S'])
                        projection_map = player['M']['projections']['M']
                        for stat in stats:
                            book_maker_pred_map = projection_map['{}'.format(
                                stat)]['M']
                            book_maker_pred_list = []
                            for book_maker in book_maker_pred_map.keys():
                                if len(book_maker_pred_map[book_maker]['L']) != 0:
                                    for idx in range(1, len(book_maker_pred_map[book_maker]['L']) + 1):
                                     if 'M' in book_maker_pred_map[book_maker]['L'][-idx]:
                                        pred = float(
                                            *book_maker_pred_map[book_maker]['L'][-idx]['M']['line'].values())
                                        book_maker_pred_list.append(pred)
                                        break
                            player_prediction[f"{stat}_external_pred"].append(
                                agg(book_maker_pred_list))
    df = pd.DataFrame()
    # if new trackers are being added make sure you add it here
    df['game_id'] = game_ids
    df['player_id'] = player_id
    for item in player_prediction.items():
        df[item[0].lower()] = item[1]
    return df


def get_prev_preds(pregame_table, stats_tracked):
    """
    Fetch DF with other our previous predictions.

    Args:
        pregame_table (List): Raw data obtained from database.
        stats_tracked (List): List of stats that we have a prediction on.

    Returns:
        df (pd.DataFrame): Dataframe with our predictions indexed on gameid and playerid
    """
    # features tracked, you can add position as required
    print("Getting prev pred")
    game_ids = []
    player_id = []
    prev_prediction = defaultdict(lambda: [])

    stats = stats_tracked
    for item in pregame_table:
        if 'teams' in item.keys():
            for team in item['teams']['L']:
                for player in team['M']['players']['L']:
                    if 'projections' in player['M']:
                        player_id.append(player['M']['id']['S'])
                        game_ids.append(item['gameid']['S'])

                        for stat in stats:
                            pred = None
                            if f'{stat}_prediction' in player['M']['projections']['M'].keys():
                                pred = float(
                                    player['M']['projections']['M'][f'{stat}_prediction']['N'])
                            prev_prediction[f'{stat}_previous'].append(pred)
    df = pd.DataFrame()
    # if new trackers are being added make sure you add it here
    df['game_id'] = game_ids
    df['player_id'] = player_id
    for item in prev_prediction.items():
        df[item[0].lower()] = item[1]
    return df


def add_external_pred(in_sample, oos, stats, df_odds):
    """
    Adds external bookmaker predictions onto samples.

    Args:
        in_sample (DataFrame): In sample.
        oos (DataFrame): List of stats that we have a prediction on.
        stats (List): List of stats to get bookmaker prediction on.
        df_odds (List): Raw data drawn from the database.
    Returns:
        in_sample_ext, oos_ext (pd.DataFrame): Dataframes with bookmaker predictions joined.
    """
    ext_pred = get_external_preds(df_odds, stats, True).dropna()
    in_sample_ext = in_sample.merge(ext_pred, how="left", on=["game_id", "player_id"]).dropna(
        subset=[x.lower()+"_external_pred" for x in stats])
    oos_ext = oos.merge(ext_pred, how="left", on=["game_id", "player_id"]).dropna(
        subset=[x.lower()+"_external_pred" for x in stats])
    return in_sample_ext, oos_ext


def add_prev_pred(in_sample, oos, stats, df_prev):
    """
    Adds our previous predictions onto samples.

    Args:
        in_sample (DataFrame): In sample.
        oos (DataFrame): List of stats that we have a prediction on.
        stats (List): List of stats to get bookmaker predictiosn on.
        df_odds (List): Raw data drawn from the database.
    Returns:
        in_sample_ext, oos_ext (pd.DataFrame): Dataframes with our previous predictions joined.
    """
    prev_pred = get_prev_preds(df_prev, stats).dropna()
    in_sample_pred = in_sample.merge(prev_pred, how="left", on=[
                                     "game_id", "player_id"]).dropna(subset=[x+"_previous" for x in stats])
    oos_pred = oos.merge(prev_pred, how="left", on=["game_id", "player_id"]).dropna(
        subset=[x+"_previous" for x in stats])
    return in_sample_pred, oos_pred


def player_metric_eval(val1, val1_name, val2, val2_name, histogram=False):
    """
    Given two lists of numbers, calculates and prints out their mse and R2 score. Optionally 
    prints a histogram of the distribution.

    Args:
        val1 (List): val1.
        val2 (List): val2.
        histogram (Bool): List of stats to get bookmaker predictiosn on.

    Returns:
        error (List): List of difference between val1 and val2 elementwise
    """
    if len(val1) == 0 or len(val2) == 0:
        print("Not enough data!")
        return None
    mse = mean_squared_error(val1, val2)
    r2 = r2_score(val1, val2)
    print(f"mse: {mse}")
    print(f"r2:  {r2}")

    if histogram:
        plt.hist(val1 - val2, bins=30)
        plt.ylabel("Amount in bin")
        plt.xlabel(f'Difference bins')
        plt.title(f'Difference between {val1_name} and {val2_name}')
        plt.show()

    error = np.array(val1 - val2)
    return error


def eval_predictions(model, stats, oos, in_sample, external_metric_eval=False, compare_to_prev_preds=False, ext_stats=[], exp_stats=[], odds_table=None, pregame_table=None, testing=False):
    """
    Trains and evaluates a lasso regression model. 

    Args:
        model (modelClass): sklearn linear model.
        stats (List): Stats to predict and evaluate on.
        oos (DataFrame): Out of sample data.
        in_sample (DataFrame): In sample data. 
        external_metric_eval (Bool): Evaluate based on external bookmaker predictions
        compare_to_prev_preds (Bool): Evaluate based on our previous predictions
        ext_stats (List): List of stats to get bookmaker prediction on.
        exp_stats (List): List of stats that we have a pervious prediction on.
        odds_table (List): Raw data of bookmaker predictions drawn from the database.
        pregame_table (List): Raw data of our previous predictions drawn from the database.
        testing (Bool): Testing?


    Returns:
        distributions (Dict): Dictionary of stats as key, and list of errors between predicted and expected values as value.
    """
    param_grid = {'model__alpha': [0.01, 0.1, 1, 10]}

    # Running the pipeline
    distributions = {}

    if external_metric_eval:
        df_odds = get_data_from_table(odds_table, testing)
        in_sample_ext, oos_ext = add_external_pred(
            in_sample, oos, ext_stats, df_odds)

    if compare_to_prev_preds:
        df_prev = get_data_from_table(pregame_table, testing)
        in_sample_prev, oos_prev = add_prev_pred(
            in_sample, oos, exp_stats, df_prev)

    for stat in stats:
        default_feature_list = [
            f'{stat}_last_2', 
            f'{stat}_last_10',
            f'std_{stat}_last_2',
            # f'{stat}_last_10_win',
            # f'{stat}_last_10_loss',
            f'{stat}_last_2_extension',
            f'{stat}_last_10_matchup',
        ]

        # Running the pipeline
        feat_coef = model_pipeline(
            in_sample, stat, model, param_grid,
            feature_selection=False,
            feature_columns_input=default_feature_list
        )

        # while len([_x for _x in feat_coef if _x[1] < 0]):
        #     # remove the feature that is getting a negative score
        #     for feat_name, coef in feat_coef:
        #         if coef < 0:
        #             print(f"REMOVING FEATURE WITH < 0 COEF: {feat_name}")
        #             default_feature_list.remove(feat_name)
        #     feat_coef = model_pipeline(
        #         in_sample, stat, model, param_grid,
        #         feature_selection=False,
        #         feature_columns_input=default_feature_list
        #     )

        print("raw coef: ", feat_coef)
        normalize_coefs = sum([_x[1] for _x in feat_coef])
        normalized_feat_coef = [(_f, _c/normalize_coefs)
                                for _f, _c in feat_coef]
        print("norm coef:", normalized_feat_coef)

        def calculate_prediction(row):
            return sum(nc*row[f] for f, nc in normalized_feat_coef)

        print(stat)
        in_sample[f'{stat}_prediction'] = in_sample.apply(
            calculate_prediction, axis=1)
        oos[f'{stat}_prediction'] = oos.apply(calculate_prediction, axis=1)

        in_sample = in_sample.dropna(
            subset=[f'{stat}_expected', f'{stat}_prediction'])
        oos = oos.dropna(subset=[f'{stat}_expected', f'{stat}_prediction'])

        print("In sample metrics:")
        player_metric_eval(
            in_sample[f"{stat}_expected"], f"{stat}_expected", in_sample[f'{stat}_prediction'], f'{stat}_prediction')

        print("OOS metrics")
        distributions[stat] = player_metric_eval(
            oos[f"{stat}_expected"], f"{stat}_expected", oos[f'{stat}_prediction'], f'{stat}_prediction', True)

        if external_metric_eval and stat in [x.lower() for x in ext_stats]:
            in_sample_ext, oos_ext = add_external_pred(
                in_sample, oos, ext_stats, df_odds)
            print("\nIn sample: ")
            print("Expected to External Prediction")
            player_metric_eval(
                in_sample_ext[f"{stat}_expected"],  f"{stat}_expected", in_sample_ext[f'{stat}_external_pred'], f'{stat}_external_pred')

            print("Internal to External Prediction")
            player_metric_eval(
                in_sample_ext[f"{stat}_prediction"], f"{stat}_prediction", in_sample_ext[f'{stat}_external_pred'], f'{stat}_external_pred')

            print("\nOOS: ")
            print("Expected to External Prediction")
            player_metric_eval(
                oos_ext[f"{stat}_expected"], f"{stat}_expected", oos_ext[f'{stat}_external_pred'],  f"{stat}_external_pred")

            print("Internal to External Prediction")
            player_metric_eval(
                oos_ext[f"{stat}_prediction"], f"{stat}_prediction", oos_ext[f'{stat}_external_pred'], f"{stat}_external_pred")

        if compare_to_prev_preds and stat in exp_stats:
            in_sample_prev, oos_prev = add_prev_pred(
                in_sample, oos, exp_stats, df_prev)
            print("\nIn sample: ")
            print("New Internal to Old Internal Prediction")
            player_metric_eval(
                in_sample_prev[f"{stat}_prediction"], f"{stat}_prediction", in_sample_prev[f'{stat}_previous'], f'{stat}_previous' + '_prediction')

            print("\nOOS: ")
            print("New Internal to Old Internal Prediction")
            player_metric_eval(
                oos_prev[f"{stat}_prediction"], f"{stat}_prediction", oos_prev[f'{stat}_previous'], f'{stat}_previous' + '_prediction')
        print("====================================================================")
    return distributions
