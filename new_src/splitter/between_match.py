# This class splits the matches themselves into a train and test-set. So an entire match is either in the train or test set. 
# This avoids data leakage between the sets. 
from pathlib import Path

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def import_csv(path):
    return pd.read_csv(path)

def main(test_size=0.2, train_size=0.8):
    csv_output_dir = os.environ.get("CSV_OUTPUT_DIR")
    csv_path = Path(csv_output_dir).resolve() / "output.csv"

    print(f"Loading CSV from {csv_path}")
    df = import_csv(csv_path)

    unique_pairs = df[["demoName", "roundIdx"]].drop_duplicates()   
    number_of_folds = int(1 / test_size)

    folds = { "csv": {}, "pkl": {}}
    for i in range(number_of_folds):
        folds["csv"][i] = []
        folds["pkl"][i] = []

    # Here we determine the random order
    unique_pairs = unique_pairs.sample(
        frac=1,
        random_state=int(os.environ.get("SPLITTING_SEED"))
    )

    fold_counter = 0
    for pair in unique_pairs.itertuples(index=False):
        folds["csv"][fold_counter].append(pair)
        folds["pkl"][fold_counter].append(pair)
        fold_counter = (fold_counter + 1) % number_of_folds

    # Uses the csv file for each fold
    for fold in folds["csv"]:
        fold_pairs = pd.DataFrame(folds["csv"][fold], columns=["demoName", "roundIdx"])
        df_fold = df.merge(fold_pairs, on=["demoName", "roundIdx"], how="inner")

        folds["csv"][fold] = df_fold

    # TODO: uses the pkl file for each fold
    for fold in folds["pkl"]:
        break

    return folds

def run():
    folds = main()
    return folds

if __name__ == "__main__":
    main()