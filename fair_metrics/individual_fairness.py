import pandas as pd
from typing import Union, Callable


def compute_conditional_probability(
    data: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    target_group: Union[str, int, float],
) -> float:
    """
    Compute the odds of a specific outcome for a target group within a dataset.

    :param data: The DataFrame containing the dataset.
    :type data: pd.DataFrame
    :param outcome_column: The name of the column in `data` representing the outcome variable.
    :type outcome_column: str
    :param group_column: The name of the column in `data` that contains the group labels.
    :type group_column: str
    :param target_group: The specific group for which the odds are to be calculated.
    :type target_group: Union[str, int, float]
    :return: The calculated odds for the target group.
    :rtype: float
    """
    positive_outcome = data[
        (data[outcome_column] == 1) & (data[group_column] == target_group)
    ].shape[0]
    outcome = data[(data[group_column] == target_group)].shape[0]
    return positive_outcome / outcome


def equalized_odds(
    data: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    feature_column: str,
    feature_func: Callable,
    reference_group: Union[str, int, float],
    target_group: Union[str, int, float],
) -> float:
    """
     Calculate the odds of a positive outcome for an individual in a specific group,
    compared to a reference group, based on a given feature.


    Parameters:
    :param data (pd.DataFrame): The dataset containing the groups, outcomes, and features.
    :param outcome_column (str): The name of the column containing the binary outcome variable.
    :param group_column (str): The name of the column containing the group labels.
    :param feature_column (str): The name of the column containing the feature based on which the odds are calculated.
    :param feature_func (Callable): A string representing the function to be applied to the feature column.
      This should be a valid function accessible in the current scope.
    :param reference_group (Union[str, int, float]): The label of the reference group for comparison.
    :param target_group (Union[str, int, float]): The label of the target group for which the odds are calculated.
    """
    filtered_df = data[data[feature_column].apply(feature_func)]
    reference_odds = compute_conditional_probability(
        filtered_df, outcome_column, group_column, reference_group
    )
    target_odds = compute_conditional_probability(
        filtered_df, outcome_column, group_column, target_group
    )
    return target_odds / reference_odds


def individial_unfairness(
    data: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    feature_column: str,
    better_feature_func: Callable,
    worse_feature_func: Callable,
    reference_group: Union[str, int, float],
    target_group: Union[str, int, float],
) -> float:
    """
    Calculate the individual unfairness in a dataset based on specified feature functions and groups.

    :param data: DataFrame containing the dataset.
    :param outcome_column: Name of the column representing the outcome variable.
    :param group_column: Name of the column containing the group labels.
    :param feature_column: Name of the column containing the feature to be transformed.
    :param better_feature_func: Callable that transforms the feature in a 'better' way.
    :param worse_feature_func: Callable that transforms the feature in a 'worse' way.
    :param reference_group: Label of the reference group for comparison.
    :param target_group: Label of the target group for comparison.
    :return: The calculated measure of individual unfairness.
    :rtype: float
    """
    better_filtered_df = data[data[feature_column].apply(better_feature_func)]
    worse_filtered_df = data[data[feature_column].apply(worse_feature_func)]
    reference_odds = compute_conditional_probability(
        worse_filtered_df, outcome_column, group_column, reference_group
    )
    target_odds = compute_conditional_probability(
        better_filtered_df, outcome_column, group_column, target_group
    )
    return target_odds / reference_odds
