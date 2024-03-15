import argparse
import pandas as pd
import torch
from train import train
from preprocess import get_original_data


def parse_args():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    with_fairness = subparser.add_parser("with_fairness")
    no_fairness = subparser.add_parser("no_fairness")

    with_fairness.add_argument("input_file", help="Reference dataframe", type=str)
    with_fairness.add_argument("S", help="Protected attribute", type=str)
    with_fairness.add_argument("Y", help="Label (decision)", type=str)
    with_fairness.add_argument(
        "underprivileged_value", help="Value for underpriviledged group", type=str
    )
    with_fairness.add_argument(
        "desired_label", help="Desired label (decision)", type=str
    )
    with_fairness.add_argument("num_epochs", help="Total number of epochs", type=int)
    with_fairness.add_argument("batch_size", help="the batch size", type=int)
    with_fairness.add_argument(
        "num_fair_epochs", help="number of fair training epochs", type=int
    )
    with_fairness.add_argument("lambda_val", help="lambda parameter", type=float)
    with_fairness.add_argument(
        "fake_name", help="name of the produced csv file", type=str
    )
    with_fairness.add_argument(
        "size_of_fake_data", help="how many data records to generate", type=int
    )

    no_fairness.add_argument("input_file", help="Reference dataframe", type=str)
    no_fairness.add_argument("num_epochs", help="Total number of epochs", type=int)
    no_fairness.add_argument("batch_size", help="the batch size", type=int)
    no_fairness.add_argument(
        "fake_name", help="name of the produced csv file", type=str
    )
    no_fairness.add_argument(
        "size_of_fake_data", help="how many data records to generate", type=int
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.command == "with_fairness":
        S = args.S
        Y = args.Y
        S_under = args.underprivileged_value
        Y_desire = args.desired_label

        df = pd.read_csv(args.input_file)

        df[S] = df[S].astype(object)
        df[Y] = df[Y].astype(object)

    elif args.command == "no_fairness":
        df = pd.read_csv(args.input_file)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")

    if args.command == "with_fairness":
        generator, critic, ohe, scaler, data_train, data_test, input_dim = train(
            df,
            args.num_epochs,
            args.batch_size,
            args.num_fair_epochs,
            args.lambda_val,
            args.command,
            S=S,
            Y=Y,
            S_under=S_under,
            Y_desire=Y_desire,
        )
    elif args.command == "no_fairness":
        generator, critic, ohe, scaler, data_train, data_test, input_dim = train(
            df, args.num_epochs, args.batch_size, 0, 0
        )
    fake_numpy_array = (
        generator(torch.randn(size=(args.size_of_fake_data, input_dim), device=device))
        .cpu()
        .detach()
        .numpy()
    )
    fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
    fake_df = fake_df[df.columns]
    fake_df.to_csv(args.fake_name, index=False)
