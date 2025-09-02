import argparse
import os

import numpy as np
import pandas as pd

DATASETS_DIR = "C:/Users/xpati/Documents/TFG/Data_Fair_MF"

parser = argparse.ArgumentParser(description="Data Splitter")
parser.add_argument("--dataset", type=str, help="Name of the dataset")
parser.add_argument("--splits", type=int, help="Number of splits")
args = parser.parse_args()

# dataset_name = args.dataset
# num_splits = args.splits

dataset_name = args.dataset if args.dataset else "ml-1m"  # default dataset
num_splits = args.splits if args.splits else 5  # default splits


def load_and_format_movielens_data(dataset_name):
    """
    Formats a raw MovieLens dataset into a the standard user/item format used in this project.

    Args:
        dataset_name (str): Name of the dataset (e.g. ml-1m)

    Returns:
        pandas.DataFrame: DataFrame containing the formatted dataset with the columns ['user_id', 'item_id', 'rating']
    """
    if dataset_name == "ml-100k":
        df = pd.read_csv(
            os.path.join(DATASETS_DIR, dataset_name, "u.data"), sep="\t", header=None
        )
    elif dataset_name == "ml-1m" or dataset_name == "ml-10m":
        df = pd.read_csv(
            os.path.join(DATASETS_DIR, dataset_name, "ratings.dat"),
            sep="::",
            engine="python",
            header=None,
        )

    df.columns = ["user_id", "item_id", "rating", "timestamp"]
    df = df[["user_id", "item_id", "rating"]]

    return df


if __name__ == "__main__":

    # Create a directory to store the train-test splits
    splits_dir = f"{DATASETS_DIR}/{dataset_name}/splits"
    os.makedirs(splits_dir, exist_ok=True)

    for i in range(1, num_splits + 1):

        # Load and format the dataset
        df = load_and_format_movielens_data(dataset_name)

        # Shuffle and split the dataset
        df = df.sample(frac=1).reset_index(drop=True)

        test_start = int(len(df) * (1 - 0.1))
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:]

        # Save the train-test splits
        train_df.to_csv(f"{splits_dir}/train_{i}.csv", index=False)
        test_df.to_csv(f"{splits_dir}/test_{i}.csv", index=False)

        print(
            f"Split {i} created with {len(train_df)} train samples and {len(test_df)} test samples."
        )

    print("All splits created successfully.")
