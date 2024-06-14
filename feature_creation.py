import pandas as pd
import numpy as np

def custom_rolling(group, stats, max_n, filter_condition=None):
    """
    Compute rolling statistics for a given group of data.

    Parameters:
    -----------
    group : pd.DataFrame
        DataFrame group on which to compute the rolling statistics.
    stats : list of str
        List of statistics to compute.
    max_n : int
        Maximum number of observations to consider.
    filter_condition : callable, optional
        A function to filter rows for the computation (default is None).

    Returns:
    --------
    dict
        Dictionary of rolling values for each statistic.
    """
    valid_stats = {stat: [] for stat in stats}
    rolling_values = {stat: {} for stat in stats}
    for _, row in group.iterrows():
        for stat in stats:
            rolling_values[stat][row.name] = valid_stats[stat][-max_n:].copy()
            if filter_condition is None or filter_condition(row):
                if stat not in row or row[stat] is None or (type(row[stat]) == list and len(row[stat]) == 0): continue
                valid_stats[stat].append(row[stat])
    return rolling_values

def compute_rolling_window(df, stats, max_n, groupby, filter_condition=None):
    """
    Compute rolling window statistics for a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame on which to compute the rolling statistics.
    stats : list of str
        List of statistics to compute.
    max_n : int
        Maximum number of observations to consider.
    groupby : list of str
        Columns to group by.
    filter_condition : callable, optional
        A function to filter rows for the computation (default is None).

    Returns:
    --------
    dict
        Dictionary of rolling window statistics.
    """
    rolling_values_series = df.groupby(groupby, sort=False).apply(lambda group: custom_rolling(group, stats, max_n, filter_condition))
    rolling_windows = {}
    for stat in stats:
        flattened_values = {}
        for group_key, stats_dict in rolling_values_series.items():
            if stat in stats_dict:
                flattened_values.update(stats_dict[stat])
        rolling_windows[stat] = pd.Series(flattened_values)

    return rolling_windows


def last_n_stats_per_player(df, stat_list, n_list=[3,5,10], use_games=True,
                            custom_suffix=None,
                            filter_condition=None, groupby=["player"]):
    """
    Compute last N stats on a per player game or avg stat per match basis

    Parameters:
    df (pandas dataframe): DataFrame with games data
    stat_list (list of str): List of column name of stat to compute e.g. ['kills', 'assists']
    n_list (list of int): List of N values to compute e.g. [5, 10]
    use_games (bool): Whether to use game level data or use match data
    custom_suffix (str or None): Custom text to append to the column names (default is None)
    match_level_avg (bool): Whether to take an avg on a per match basis or return raw value, default is True
    filter_condition (func): A function that gets applied to the row if we want to keep it (wins/loss)
    groupby (list of str): tells what to groupby (we need to change for matchup and so forth)
    Returns:
    df_modified: Modified DataFrame with additional columns:
        {stat}_last_{n}: Stat value for last n games
    """
    # Input validation
    if len(df) == 0:
        return pd.DataFrame()

    df_modified = df.copy()
    # Compute optimized rolling window
    max_n = max(n_list)
    rolling_window_mapping = compute_rolling_window(df, stat_list, max_n, groupby ,filter_condition)
    # Compute last n stats for each N
    for stat in stat_list:
        rolling_window = rolling_window_mapping[stat]
        for n in n_list:
            # Flatten the list if using games. Takes n newest games, then takes n newest matches
            if use_games:
                last_n_stats = rolling_window.apply(lambda x: [item for sublist in x for item in sublist][-n:])
            # If not using games and also not calculating match-level averages
            else:
                last_n_stats = rolling_window.apply(lambda x: x[-n:]) #Takes n newest games, then averages them.
                last_n_stats = last_n_stats.apply(lambda x: [sum(sublist) / len(sublist) if (type(sublist) == list and len(sublist) != 0) else sublist for sublist in x]) 
            stat_suffix = f'{stat}_last_{n}'
            stat_std_suffix = f'std_{stat}_last_{n}'
            if custom_suffix:
                stat_suffix = f'{stat_suffix}_{custom_suffix}'
                stat_std_suffix = f'{stat_std_suffix}_{custom_suffix}'
            df_modified[f'{stat_suffix}'] = last_n_stats
            df_modified[f'{stat_std_suffix}'] = df_modified[f'{stat_suffix}'].apply(lambda row: np.std(list(map(float, row))) if len(row) > 1 else np.nan)
    return df_modified

def filter_wins(row):
    if 'win_result' not in row: return False
    return int(row['win_result']) == 1

def filter_losses(row):
    if 'win_result' not in row: return False
    return  int(row['win_result']) == 0

def generate_target(df, stat_list, use_median=True):
    """
    Generate target statistics for a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame on which to generate the target statistics.
    stat_list : list of str
        List of statistics to generate.
    use_median : bool, optional
        Whether to use the median or mean (default is True).

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns for the expected values of the specified statistics.
    """
    for stat in stat_list:
        if stat == "win_result": use_median = False
        if use_median:
            df[f"{stat}_expected"] = df[stat].apply(lambda row: np.median(row)).round(2)
        else:
            df[f"{stat}_expected"] = df[stat].apply(lambda row: np.mean(list(map(float, row)))).round(2)
    return df

