import pandas as pd
from typing import Union, Callable
from sklearn.metrics import accuracy_score


def disparity_ratio(
    data: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    reference_group: Union[str, int, float],
    target_group: Union[str, int, float],
) -> float:
    """
    Calculate the Disparity Ratio between two groups for a specific outcome.

    :param data: Pandas DataFrame containing the outcome and group data
    :param outcome_column: The name of the column with the outcome data
    :param group_column: The name of the column with the group labels
    :param reference_group: The label of the reference group in the group column
    :param target_group: The label of the target group in the group column
    :return: Disparity Ratio of the target group compared to the reference group
    """
    reference_rate = data[data[group_column] == reference_group][outcome_column].mean()
    target_rate = data[data[group_column] == target_group][outcome_column].mean()

    return target_rate / reference_rate


def social_benefit(
    data: pd.DataFrame,
    outcome_column: str,
    prediction_fun: Callable,
) -> float:
    """
    Calculate the social benefit of a prediction model applied to a transformed feature of a dataset.

    :param data: DataFrame containing the dataset.
    :param prediction_fun: Callable prediction function that takes the transformed feature and
                           returns a prediction.
    :return: The calculated measure of social benefit.
    :rtype: float
    """
    outcomes = data[outcome_column]
    featured_data = data.drop(columns=[outcome_column])
    predictions = prediction_fun(featured_data)
    return accuracy_score(outcomes, predictions)
