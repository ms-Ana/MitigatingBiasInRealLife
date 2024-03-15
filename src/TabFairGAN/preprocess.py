import pandas as pd
import numpy as np
import torch
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


def get_ohe_data_with_fairness(df, S, Y, S_under, Y_desire):
    df_int = df.select_dtypes(["float", "integer"]).values
    continuous_columns_list = list(df.select_dtypes(["float", "integer"]).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution="uniform")
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes("object")
    df_cat_names = list(df.select_dtypes("object").columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    S_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[
            : list(discrete_columns_ordereddict.keys()).index(S)
        ]
    )
    Y_start_index = len(continuous_columns_list) + sum(
        list(discrete_columns_ordereddict.values())[
            : list(discrete_columns_ordereddict.keys()).index(Y)
        ]
    )

    if (
        ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0]
        == S_under
    ):
        underpriv_index = 0
        priv_index = 1
    else:
        underpriv_index = 1
        priv_index = 0
    if (
        ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0]
        == Y_desire
    ):
        desire_index = 0
        undesire_index = 1
    else:
        desire_index = 1
        undesire_index = 0

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return (
        ohe,
        scaler,
        discrete_columns_ordereddict,
        continuous_columns_list,
        final_array,
        S_start_index,
        Y_start_index,
        underpriv_index,
        priv_index,
        undesire_index,
        desire_index,
    )


def get_ohe_data_no_fairness(df):
    df_int = df.select_dtypes(["float", "integer"]).values
    continuous_columns_list = list(df.select_dtypes(["float", "integer"]).columns)
    ##############################################################
    scaler = QuantileTransformer(n_quantiles=2000, output_distribution="uniform")
    df_int = scaler.fit_transform(df_int)

    df_cat = df.select_dtypes("object")
    df_cat_names = list(df.select_dtypes("object").columns)
    numerical_array = df_int
    ohe = OneHotEncoder()
    ohe_array = ohe.fit_transform(df_cat)

    cat_lens = [i.shape[0] for i in ohe.categories_]
    discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

    final_array = np.hstack((numerical_array, ohe_array.toarray()))
    return (
        ohe,
        scaler,
        discrete_columns_ordereddict,
        continuous_columns_list,
        final_array,
    )


def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[
        :, : df_orig.select_dtypes(["float", "integer"]).shape[1]
    ]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[
        :, df_orig.select_dtypes(["float", "integer"]).shape[1] :
    ]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(
        df_ohe_int, columns=df_orig.select_dtypes(["float", "integer"]).columns
    )
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes("object").columns)
    return pd.concat([df_int, df_cat], axis=1)


def prepare_data_with_fairness(df, batch_size, **kwargs):
    (
        ohe,
        scaler,
        discrete_columns,
        continuous_columns,
        df_transformed,
        S_start_index,
        Y_start_index,
        underpriv_index,
        priv_index,
        undesire_index,
        desire_index,
    ) = get_ohe_data_with_fairness(
        df, kwargs["S"], kwargs["Y"], kwargs["S_under"], kwargs["Y_desire"]
    )
    input_dim = df_transformed.shape[1]
    X_train, X_test = train_test_split(df_transformed, test_size=0.1, shuffle=True)
    data_train = X_train.copy()
    data_test = X_test.copy()

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    data = torch.from_numpy(data_train).float()

    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    return (
        ohe,
        scaler,
        input_dim,
        discrete_columns,
        continuous_columns,
        train_dl,
        data_train,
        data_test,
        S_start_index,
        Y_start_index,
        underpriv_index,
        priv_index,
        undesire_index,
        desire_index,
    )


def prepare_data_no_fairness(df, batch_size):
    # df = pd.concat([df_train, df_test], axis=0)

    (
        ohe,
        scaler,
        discrete_columns,
        continuous_columns,
        df_transformed,
    ) = get_ohe_data_no_fairness(df)

    input_dim = df_transformed.shape[1]

    # from sklearn.model_selection import train_test_split
    #################
    X_train, X_test = train_test_split(
        df_transformed, test_size=0.1, shuffle=True
    )  # random_state=10)
    # X_train = df_transformed[:df_train.shape[0],:]
    # X_test = df_transformed[df_train.shape[0]:,:]

    data_train = X_train.copy()
    data_test = X_test.copy()

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    data = torch.from_numpy(data_train).float()

    train_ds = TensorDataset(data)
    train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    return (
        ohe,
        scaler,
        input_dim,
        discrete_columns,
        continuous_columns,
        train_dl,
        data_train,
        data_test,
    )
