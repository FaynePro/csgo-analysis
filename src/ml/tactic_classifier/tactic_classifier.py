from ml.tactic_classifier.classifier_trainer.single_split_trainer import SingleSplitTrainer
from ml.tactic_classifier.classifier_trainer.kfold_trainer import KfoldTrainer
import argparse

# TODO: Retrieve file paths from environment variables instead of hardcoding

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess CS:GO/CS2 demo files into raw frame and player data."
    )

    parser.add_argument("--kfold", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO"
    )

    parser.add_argument(
        "--data-root-dir",
        type=str,
        default="data/preprocessed/de_dust2",
        help="Path to preprocessed data directory"
    )

    parser.add_argument(
        "--tactics-json-path",
        type=str,
        default="data/tactic_labels/de_dust2_tactics.json",
        help="Path to tactics labels JSON file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Path to output directory for results (used for K-Fold training)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.kfold:
        print("Running K-Fold training...")
        trainer = KfoldTrainer()
        model, dataset, fold_results, summary, test_metrics = trainer.train_classifier_kfold(
            data_root_dir=args.data_root_dir,
            tactics_json_path=args.tactics_json_path,
            num_epochs=50,
            k_folds=10,
            output_dir=args.output_dir
        )
    else:
        print("Running single split training...")
        trainer = SingleSplitTrainer()
        model, dataset, history = trainer.train_classifier(
            data_root_dir=args.data_root_dir,
            tactics_json_path=args.tactics_json_path,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
