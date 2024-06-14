import pandas as pd

def weighted_average(data_points, decaying_weight):
    """
    Calculate the weighted average of a list of data points using a decaying weight factor.

    Args:
        data_points (list): List of numeric data points.
        decaying_weight (float): Decaying weight factor for weighting data points.

    Returns:
        float: Weighted average of the data points.
    """
    weight = 1.0  # Initial weight for the newest data point
    weighted_sum = 0.0
    total_weight = 0.0
    for value in reversed(data_points):
        weighted_sum += float(value) * weight
        total_weight += weight
        weight *= decaying_weight  # Decrease the weight by X% for the next data point

    if total_weight == 0:
        return 0  # To avoid division by zero
    else:
        return weighted_sum / total_weight

def stat_aggregator(df, stat_list, decaying_n_stat_weight=1, decaying_match_stat_weight=1):
    """
    Calculate weighted averages for specified statistics in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing statistical data.
        stat_list (list): List of statistics to be aggregated.
        decaying_weight (float, optional): Decaying weight factor for weighting data points. Default is 1.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics.
    """
    df_temp = df.copy()
    for stat in stat_list:
        list_of_columns = [column_name for column_name in df_temp.columns if stat in column_name and "expected" not in column_name and "std" not in column_name and "extension" not in column_name]
        print(list_of_columns)
        df_temp[list_of_columns] = pd.concat([df_temp[[list_of_columns[0]]].applymap(lambda x: weighted_average(x, decaying_match_stat_weight)), df_temp[list_of_columns[1:]].applymap(lambda x: weighted_average(x, decaying_n_stat_weight))], axis = 1)

    return df_temp
