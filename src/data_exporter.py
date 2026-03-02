"""
Exports preprocessed demo data to CSV or pickle format.
Handles flattening of nested frame/round/player data structures for tabular output.

Usage:
    python data_exporter.py preprocessed_file.pkl --output-type csv
    python data_exporter.py preprocessed_file.pkl --output-type both --output-dir data/exported
    python data_exporter.py preprocessed_file.pkl --round 0
"""

import csv
import pickle
from pathlib import Path
from typing import Any, Optional
import logging

from preprocessing.functions.data_preprocessor import KEYS_PLAYER_LEVEL, KEYS_ROUND_LEVEL, KEYS_FRAME_LEVEL


def flatten_preprocessed_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten preprocessed frame into a single-level dictionary for CSV export.
    
    Args:
        frame: Preprocessed frame dict with nested structures
        
    Returns:
        Flattened dictionary with all values at top level
    """
    flattened = {}
    
    # Add round-level data
    if "round_data" in frame:
        for key, value in frame["round_data"].items():
            flattened[f"round_{key}"] = value
    
    # Add frame-level data
    if "frame_data" in frame:
        for key, value in frame["frame_data"].items():
            flattened[f"frame_{key}"] = value
    
    # Add map and tactic
    flattened["map_name"] = frame.get("map_name", "")
    flattened["tactic"] = frame.get("tactic", "unknown")
    
    # Add bomb data
    if "bomb_data" in frame:
        for key, value in frame["bomb_data"].items():
            flattened[f"bomb_{key}"] = value
    
    # Add player data (flatten each player's attributes with player index)
    if "players_data" in frame:
        for player_idx, player_data in enumerate(frame["players_data"]):
            for key, value in player_data.items():
                flattened[f"player{player_idx}_{key}"] = value
    
    return flattened


def get_csv_headers() -> list[str]:
    """Generate CSV headers for flattened preprocessed data."""
    headers = []
    
    # Round-level headers
    for key in KEYS_ROUND_LEVEL:
        headers.append(f"round_{key}")
    
    # Frame-level headers
    for key in KEYS_FRAME_LEVEL:
        headers.append(f"frame_{key}")
    
    # Map and tactic
    headers.append("map_name")
    headers.append("tactic")
    
    # Bomb data (estimated 10 fields from bomb info)
    bomb_keys = ["x", "y", "z", "velocityX", "velocityY", "velocityZ", 
                 "isPlanted", "bombAreaId", "plantedSite", "isDefusing"]
    for key in bomb_keys:
        headers.append(f"bomb_{key}")
    
    # Player data (5 players max, with all player-level attributes)
    for player_idx in range(5):
        for key in KEYS_PLAYER_LEVEL:
            headers.append(f"player{player_idx}_{key}")
    
    return headers

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess CS:GO/CS2 demo files into raw frame and player data."
    )

    parser = argparse.ArgumentParser(
        description="Export preprocessed frame data to pickle or CSV format."
    )
    parser.add_argument(
        "input_file",
        help="Path to preprocessed pickle file to export",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exported",
        help="Output directory where exported files will be saved (default: data/exported)",
    )
    parser.add_argument(
        "--output-type",
        choices=["pickle", "csv", "both"],
        default="both",
        help="Output format: 'pickle', 'csv', or 'both' (default: both)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Export only specific round index (default: export all rounds)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()

class DataExporter:
    """Exports preprocessed data to pickle or CSV format."""
    
    def __init__(self, output_dir: Path, logger=None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory where processed data will be saved
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def export_to_pickle(
        self,
        match_id: str,
        preprocessed_rounds: dict[int, list[dict[str, Any]]],
        round_idx: Optional[int] = None,
    ) -> Path:
        """
        Export preprocessed data to pickle format.
        
        Args:
            match_id: Match ID for creating subdirectory
            preprocessed_rounds: Dict mapping round index to list of preprocessed frames
            round_idx: If provided, only export this specific round; else export all rounds
            
        Returns:
            Path to output file
        """
        match_dir = self.output_dir / match_id
        match_dir.mkdir(parents=True, exist_ok=True)
        
        if round_idx is not None:
            # Export single round
            output_path = match_dir / f"exported-demo-round-{round_idx}.pkl"
            data = preprocessed_rounds.get(round_idx, [])
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            self.logger.info(f"Exported round {round_idx} to {output_path}")
        else:
            # Export all rounds
            output_path = match_dir / "exported-demo.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(preprocessed_rounds, f)
            self.logger.info(f"Exported all rounds to {output_path}")
        
        return output_path
    
    def export_to_csv(
        self,
        match_id: str,
        preprocessed_rounds: dict[int, list[dict[str, Any]]],
        round_idx: Optional[int] = None,
    ) -> Path:
        """
        Export preprocessed data to CSV format.
        
        Args:
            match_id: Match ID for creating subdirectory
            preprocessed_rounds: Dict mapping round index to list of preprocessed frames
            round_idx: If provided, only export this specific round; else export all rounds
            
        Returns:
            Path to output file
        """
        match_dir = self.output_dir / match_id
        match_dir.mkdir(parents=True, exist_ok=True)
        
        headers = get_csv_headers()
        
        if round_idx is not None:
            # Export single round
            output_path = match_dir / f"exported-demoo-round-{round_idx}.csv"
            frames = preprocessed_rounds.get(round_idx, [])
        else:
            # Export all rounds
            output_path = match_dir / "exported-demo.csv"
            frames = []
            for round_data in preprocessed_rounds.values():
                frames.extend(round_data)
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for frame in frames:
                flattened = flatten_preprocessed_frame(frame)
                # Ensure all headers are present in row (fill missing with empty strings)
                row = {header: flattened.get(header, "") for header in headers}
                writer.writerow(row)
        
        self.logger.info(f"Exported {len(frames)} frames to {output_path}")
        return output_path
    
    def export(
        self,
        match_id: str,
        preprocessed_rounds: dict[int, list[dict[str, Any]]],
        output_type: str = "pickle",
        round_idx: Optional[int] = None,
    ) -> Path:
        """
        Export preprocessed data in specified format.
        
        Args:
            match_id: Match ID for creating subdirectory
            preprocessed_rounds: Dict mapping round index to list of preprocessed frames
            output_type: "pickle" or "csv"
            round_idx: If provided, only export this specific round; else export all rounds
            
        Returns:
            Path to output file
        """
        if output_type == "pickle":
            return self.export_to_pickle(match_id, preprocessed_rounds, round_idx)
        elif output_type == "csv":
            return self.export_to_csv(match_id, preprocessed_rounds, round_idx)
        else:
            raise ValueError(f"Unsupported output type: {output_type}")


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path
    
    args = parse_args()
    
    # Setup logging
    logging_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Load preprocessed data
    logger.info(f"Loading preprocessed data from {args.input_file}")
    with open(args.input_file, "rb") as f:
        preprocessed_rounds = pickle.load(f)
    
    # Extract match ID from filename or use timestamp
    input_path = Path(args.input_file)
    match_id = input_path.parent.name if input_path.parent != input_path.parent.parent else "match_" + str(int(time.time()))
    
    # Create exporter
    exporter = DataExporter(Path(args.output_dir), logger=logger)
    
    # Export
    if args.output_type == "both":
        logger.info("Exporting to both pickle and CSV formats...")
        exporter.export_to_pickle(match_id, preprocessed_rounds, args.round)
        exporter.export_to_csv(match_id, preprocessed_rounds, args.round)
    else:
        exporter.export(match_id, preprocessed_rounds, args.output_type, args.round)
    
    logger.info("Export complete")
