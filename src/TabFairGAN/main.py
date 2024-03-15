import argparse
import pandas as pd
import torch
from train import train
from preprocess import get_original_data


TRAIN_CONFIG = {
    "command": "with_fairness",
    "S": "sex",  # Protected attirbute
    "Y": "income",  # Label
    "underprivileged_value": "Female",
    "desired_label": 1.0,
    "num_epochs": 200,
    "batch_size": 256,
    "num_fair_epochs": 30,
    "lambda_val": 0.5,
    "save_file": "fake_adult.csv",
    "size_of_fake_data": 32561,  # how many data records to generate
    "input_file": "/home/ana/University/MitigatingBiasInDataScience/processed_data/adult.csv",
}

if __name__ == "__main__":
    if TRAIN_CONFIG["command"] == "with_fairness":
        S = TRAIN_CONFIG["S"]
        Y = TRAIN_CONFIG["Y"]
        S_under = TRAIN_CONFIG["underprivileged_value"]
        Y_desire = TRAIN_CONFIG["desired_label"]

        df = pd.read_csv(TRAIN_CONFIG["input_file"])

        df[S] = df[S].astype(object)
        df[Y] = df[Y].astype(object)

    elif TRAIN_CONFIG["command"] == "no_fairness":
        df = pd.read_csv(TRAIN_CONFIG["input_file"])

    device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

    if TRAIN_CONFIG["command"] == "with_fairness":
        generator, critic, ohe, scaler, data_train, data_test, input_dim = train(
            df,
            TRAIN_CONFIG["num_epochs"],
            TRAIN_CONFIG["batch_size"],
            TRAIN_CONFIG["num_fair_epochs"],
            TRAIN_CONFIG["lambda_val"],
            TRAIN_CONFIG["command"],
            S=S,
            Y=Y,
            S_under=S_under,
            Y_desire=Y_desire,
        )
    elif TRAIN_CONFIG["command"] == "no_fairness":
        generator, critic, ohe, scaler, data_train, data_test, input_dim = train(
            df, TRAIN_CONFIG["num_epochs"], TRAIN_CONFIG["batch_size"], 0, 0
        )
    fake_numpy_array = (
        generator(
            torch.randn(
                size=(TRAIN_CONFIG["size_of_fake_data"], input_dim), device=device
            )
        )
        .cpu()
        .detach()
        .numpy()
    )
    fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
    fake_df = fake_df[df.columns]
    fake_df.to_csv(TRAIN_CONFIG["save_file"], index=False)
