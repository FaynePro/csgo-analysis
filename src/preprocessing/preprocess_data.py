import argparse
import logging
from dotenv import load_dotenv
from preprocessing.functions.data_preprocessor import process_single_file
from preprocessing.functions.preprocess_demo_batch import process_batch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess CS:GO/CS2 demo files into raw frame and player data."
    )

    parser.add_argument("demo_path", nargs="?", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--tactic-labels-dir", default="data/tactic_labels")
    parser.add_argument("--output-dir", default="data/preprocessed")
    parser.add_argument("--batch-file-names", default="CREATE_GRAPHS_FILENAMES_PATH")
    parser.add_argument("--batch-files-path", default="CREATE_GRAPHS_DEMO_DIR")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--reprocess", action="store_true")

    return parser.parse_args()

def main():
    load_dotenv()
    
    args = parse_args()
    
    # Setup logging
    logging_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Single file mode
    if args.demo_path and not args.batch:
        process_single_file(args, logger)
    
    # Batch mode
    elif args.batch or not args.demo_path:
        process_batch(args, logger)


if __name__ == "__main__":
    main()
