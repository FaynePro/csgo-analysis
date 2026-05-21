# This class splits the matches themselves into a train and test-set. So an entire match is either in the train or test set. 
# This avoids data leakage between the sets. 
from pathlib import Path

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def import_csv(path):
    return pd.read_csv(path)

# TODO: Currently a placeholder, I will implement this splitting alternative some other time
def main(test_size=0.2, train_size=0.8):
    return None

def run():
    folds = main()
    return folds

if __name__ == "__main__":
    main()