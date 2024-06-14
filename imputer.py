import pandas as pd
import numpy as np
import json
import re

class Imputer:
    """
    A class used to impute missing values in a dataset based on various levels of grouping.

    ...

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to be imputed.
    stat_columns : list
        The columns which contain the statistics to be imputed.
    use_median : bool, optional
        A flag to toggle between using median or mean for imputation (default is True).
    role : str, optional
        The role to consider for role-based imputation (default is None).
    tournament_imputation : bool, optional
        A flag to enable tournament-based imputation (default is False).
    blacklist_tournaments : list, optional
        A list of tournament IDs to exclude from tournament-based imputation (default is None).
    statistics_cache : dict
        A cache to store the computed statistics used for imputation.

    Methods
    -------
    calculate_statistics():
        Calculates and caches the statistics used for imputation.
    impute_stat(stat, player_team_id, opposing_team_id, tournament_id):
        Imputes a single statistic based on various levels of grouping.
    fill_missing_values():
        Fills missing values in the DataFrame based on the cached statistics.
    convert_to_json_serializable():
        Converts the statistics cache to a JSON-serializable format.
    """

    def __init__(self, df, stat_columns, use_median=True, role=None,
                 tournament_imputation=False, blacklist_tournaments=None):
        self.df = df.copy()
        self.stat_columns = stat_columns
        self.use_median = use_median
        self.role = role
        self.tournament_imputation = tournament_imputation
        self.blacklist_tournaments = blacklist_tournaments if blacklist_tournaments else []
        self.statistics_cache = {}
        self.calculate_statistics()

    def calculate_statistics(self):
        agg_func = 'median' if self.use_median else 'mean'
        stat_names_list = set()
        for stat_name in self.stat_columns:
            stat_temp = stat_name.split('_')[0]
            stat = f"{stat_temp}_expected"
            stat_names_list.add(stat)
        stat_names_list = list(stat_names_list)
        for stat in stat_names_list:
            if "std" in stat:
                continue    
            stat_cache = {
                'global': {},
                'team': {},
                'matchup': {},
                'tournament': {},
            }
            # Global role-based statistic
            if self.role:
                stat_cache['global']['role'] = self.df[self.df['role'] == self.role][stat].agg(agg_func)
            stat_cache['global']['all'] = self.df[stat].agg(agg_func)
            # Global statistic
            grouping_params = ['team_id', 'role'] if self.role else ['team_id']
            counts = self.df.groupby(grouping_params).size()
            sufficient_data_teams = counts[counts >= 500 if stat != 'winresult' else 500]
            filtered_df = self.df.set_index(grouping_params).loc[sufficient_data_teams.index]
            stat_cache['team'] = filtered_df.groupby(grouping_params)[stat].agg(agg_func)
            grouping_params = ['team_id', 'opposing_id', 'role'] if self.role else ['team_id', 'opposing_id']
            matchup_counts = self.df.groupby(grouping_params).size()
            sufficient_data_matchups = matchup_counts[matchup_counts > 0 if stat != "winresult" else 50]
            filtered_matchup_df = self.df.set_index(grouping_params).loc[sufficient_data_matchups.index]
            stat_cache['matchup'] = filtered_matchup_df.groupby(grouping_params)[stat].agg(agg_func)
            if self.tournament_imputation:
                grouping_params = ['tournament_id', 'role'] if self.role else ['tournament_id']
                tournament_counts = self.df.groupby(grouping_params).size()
                sufficient_data_tournaments = tournament_counts[tournament_counts >= 500 if stat != "winresult" else 500]
                filtered_tournament_df = self.df.set_index(grouping_params).loc[sufficient_data_tournaments.index]
                stat_cache['tournament'] = filtered_tournament_df.groupby(grouping_params)[stat].agg(agg_func)
            self.statistics_cache[stat] = stat_cache

    def impute_stat(self, stat, player_team_id, opposing_team_id, tournament_id):
        """
        Imputes a single statistic based on various levels of grouping.

        Parameters:
        -----------
        stat : str
            The statistic to be imputed.
        player_team_id : str
            The ID of the player's team.
        opposing_team_id : str
            The ID of the opposing team.
        tournament_id : str
            The ID of the tournament.

        Returns:
        --------
        float
            The imputed value for the specified statistic.
        """
        cache = self.statistics_cache[stat]
        grouping_params = (player_team_id, opposing_team_id, self.role) if self.role else (player_team_id, opposing_team_id)
        if grouping_params in cache['matchup'].index:
            return cache['matchup'][grouping_params]
        grouping_params = (player_team_id, self.role) if self.role else player_team_id
        if grouping_params in cache['team'].index:
            return cache['team'][grouping_params]
        if self.tournament_imputation and tournament_id not in self.blacklist_tournaments:
            grouping_params = (tournament_id, self.role) if self.role else tournament_id
            if grouping_params in cache['tournament'].index:
                return cache['tournament'][grouping_params]
        return cache['global']['role'] if self.role and 'role' in cache['global'] else cache['global']['all']

    def fill_missing_values(self):
        """Fills missing values in the DataFrame based on the cached statistics."""
        for stat in self.stat_columns:
            if "std" in stat:
                mean = np.nanmean(list(self.df[stat]))
                self.df[stat] = self.df[stat].fillna(mean)
            else:
                stat_temp = stat.split('_')[0]
                stat_target_name = f"{stat_temp}_expected"
                match = re.search(r'\d+', stat)
                expected_length = int(match.group())
                self.df[f'{stat}_extension'] = self.df.apply(
                    lambda x: self.impute_stat(stat_target_name, x['team_id'], x['opposing_id'], x['tournament_id']),
                    axis=1
                )
                mask = self.df[stat].apply(len) < expected_length
                self.df.loc[mask, stat] = self.df.loc[mask].apply(
                    lambda row: [row[f'{stat}_extension']] * (expected_length - len(row[stat])) + row[stat],
                    axis=1
                )
                #self.df.drop(f'{stat}_extension', axis=1, inplace=True)
        stat_columns = [stat for stat in self.stat_columns if "std" not in stat]
        missing_mask = self.df[stat_columns].applymap(lambda cell: any(x is None for x in cell)).any(axis=1)
        missing_indices = self.df[missing_mask].index
        for idx in missing_indices:
            player_team_id = self.df.at[idx, 'team_id']
            opposing_team_id = self.df.at[idx, 'opposing_id']
            tournament_id = self.df.at[idx, 'tournament_id']
            for stat in stat_columns:
                stat_temp = stat.split('_')[0]
                stat_target_name = f"{stat_temp}_expected"
                if any(pd.isnull(self.df.at[idx, stat])):
                    self.df.at[idx, stat] = ([self.impute_stat(stat_target_name, player_team_id, opposing_team_id, tournament_id)] * sum(x is None for x in self.df.at[idx, stat]) 
                                            + list(filter(lambda val: val is not None, self.df.at[idx, stat])))
        return self.df

    def convert_to_json_serializable(self):
        """
        Converts the statistics cache to a JSON-serializable format.

        Returns:
        --------
        dict
            The statistics cache in a JSON-serializable format.
        """
        json_serializable_cache = {}
        for stat, caches in self.statistics_cache.items():
            clean_stat = stat.replace('_expected', '')
            json_serializable_cache[clean_stat] = {}
            for cache_type, cache in caches.items():
                if isinstance(cache, pd.Series):
                    nested_dict = {}
                    for index, value in cache.items():
                        keys = index if isinstance(index, tuple) else (index,)
                        current_dict = nested_dict
                        for key in keys[:-1]:
                            current_dict = current_dict.setdefault(key, {})
                        current_dict[keys[-1]] = value
                    json_serializable_cache[clean_stat][cache_type] = nested_dict
                elif isinstance(cache, pd.DataFrame):
                    # Handle DataFrame case if needed
                    pass
                elif isinstance(cache, dict):
                    json_serializable_cache[clean_stat][cache_type] = cache
                else:
                    json_serializable_cache[clean_stat][cache_type] = cache.tolist() if hasattr(cache, 'tolist') else cache
        return json_serializable_cache
    
    def fill_centered_std(self, df):
        """
        Adds centered std to the table
        Parameters:
        -----------
        df : DataFrame
            DataFrame to be filled.

        Returns:
        --------
        df : DataFrame
            DataFrame with filled centered std
        """
        matching_columns = []
        for column in df.columns:
            if "std" in column:
                matching_columns.append(column)
        for column in matching_columns:
            std = np.nanmean(list(df[column]))
            df[f'centered_{column}'] = df[column].apply(lambda x: np.round(x - std, 3))
        return df