"""
Loads demo files and extracts raw frame & round data (before graph transformation).
Handles demo validation, frame extraction, and tactic label loading.

Usage:
    Single file:
        python data_preprocessor.py demo.dem
    
    Batch (reads from JSON):
        python data_preprocessor.py --batch
"""

import json
import logging
import pickle
import stats

from pathlib import Path
from typing import Any, Optional
from datamodel.data_manager import DataManager
from utils.project_root import find_project_root

KEYS_ROUND_LEVEL = (
    "roundNum",
    "isWarmup",
    "winningSide",
    "losingTeam",
    "tFreezeTimeEndEqVal",
    "tRoundStartEqVal",
    "tRoundSpendMoney",
)

KEYS_FRAME_LEVEL = ("tick", "seconds", "bombPlanted")

KEYS_PLAYER_LEVEL = (
    "x",
    "y",
    "z",
    "velocityX",
    "velocityY",
    "velocityZ",
    "viewX",
    "viewY",
    "hp",
    "armor",
    "activeWeapon",
    "totalUtility",
    "isAlive",
    "isDefusing",
    "isPlanting",
    "isReloading",
    "isInBombZone",
    "isInBuyZone",
    "equipmentValue",
    "equipmentValueFreezetimeEnd",
    "equipmentValueRoundStart",
    "cash",
    "cashSpendThisRound",
    "cashSpendTotal",
    "hasHelmet",
    "hasDefuse",
    "hasBomb",
)

WEAPON_ID_MAPPING = {
    "": 0,
    "Decoy Grenade": 1,
    "AK-47": 2,
    "M4A1": 3,
    "Incendiary Grenade": 4,
    "Knife": 5,
    "MAC-10": 6,
    "USP-S": 7,
    "Tec-9": 8,
    "AWP": 9,
    "Glock-18": 10,
    "SSG 08": 11,
    "HE Grenade": 12,
    "Galil AR": 13,
    "C4": 14,
    "Smoke Grenade": 15,
    "Molotov": 16,
    "P250": 17,
    "Flashbang": 18,
    "SG 553": 19,
    "Desert Eagle": 20,
    "Zeus x27": 21,
    "CZ75 Auto": 22,
    "M4A4": 23,
    "Five-SeveN": 24,
    "AUG": 25,
    "FAMAS": 26,
    "MP9": 27,
    "G3SG1": 28,
    "UMP-45": 29,
    "MP5-SD": 30,
    "Dual Berettas": 31,
    "P2000": 32,
    "MP7": 33,
    "Nova": 34,
    "XM1014": 35,
    "MAG-7": 36,
    "Sawed-Off": 37,
    "SCAR-20": 38,
    "PP-Bizon": 39,
    "M249": 40,
    "Negev": 41,
    "Taser": 42,
    "R8 Revolver": 43,
    "M4A1-S": 44,
}

# def preprocess_round(
#     dm: DataManager,
#     round_idx: int,
#     frame_tactic_map: Optional[dict[str, str]] = None,
#     logger=None,
#     strict: bool = False,
# ) -> list[dict[str, Any]]:
#     """
#     Preprocess a single round: extract raw frame and player data.
    
#     Returns a list of preprocessed frame data dictionaries, one per valid frame.
#     Each dictionary contains:
#     - round_data: Round-level metadata
#     - frame_data: Frame-level metadata (tick, seconds, bombPlanted)
#     - team_data: All players on terrorist side with their attributes
#     - bomb_data: Bomb position and state
#     - tactic: Tactic label for this frame
#     """
#     round_obj = dm.get_game_round(round_idx)
#     map_name = dm.get_map_name()

#     # All variables on the round level
#     round_data = {key: round_obj[key] for key in KEYS_ROUND_LEVEL}

#     frames = dm._get_frames(round_idx)
#     logger.info("Preprocessing round %d with %d frames." % (round_idx, len(frames)))

#     # Store crucial bomb events for later analysis
#     bomb_event_data = stats.process_bomb_data(round_obj)

#     # Preprocess each frame
#     preprocessed_frames = []
#     error_frame_count = 0
#     total_frames = len(frames)

#     for frame_idx, frame in enumerate(frames):
#         # Check validity of frame
#         valid_frame, err_text = stats.check_frame_validity(frame)
#         if not valid_frame:
#             logger.warning(f"Frame {frame_idx} skipped: {err_text}")
#             if strict:
#                 raise ValueError(f"Invalid frame {frame_idx}: {err_text}")
#             error_frame_count += 1
#             continue

#         # Tactic label for this frame
#         tactic = (
#             frame_tactic_map.get(str(frame_idx), "unknown")
#             if frame_tactic_map
#             else "unknown"
#         )

#         # All variables on the frame level
#         frame_data = {key: frame[key] for key in KEYS_FRAME_LEVEL}

#         # Adjust seconds with bomb event timing if applicable
#         if (
#             bomb_event_data["bombTick"] is not None
#             and frame["tick"] >= bomb_event_data["bombTick"]
#         ):
#             frame_data["seconds"] = frame["seconds"] + bomb_event_data["bombSeconds"]
#         else:
#             frame_data["seconds"] = frame["seconds"]

#         # Extract team (T side) data
#         team = frame["t"]

#         # Process all players on the team
#         players_data = []
#         for player in sorted(
#             team["players"],
#             key=lambda p: dm.get_player_idx_mapped(p["name"], "t", frame),
#         ):
#             player_data = {key: player[key] for key in KEYS_PLAYER_LEVEL}
#             player_data["activeWeapon"] = map_weapon_to_id(
#                 player_data["activeWeapon"], logger=logger
#             )
#             players_data.append(player_data)

#         # Get bomb data
#         bomb_data = dm.get_bomb_info(round_idx, frame_idx)

#         # Compile preprocessed frame
#         preprocessed_frame = {
#             "round_data": round_data,
#             "frame_data": frame_data,
#             "players_data": players_data,
#             "bomb_data": bomb_data,
#             "tactic": tactic,
#             "map_name": map_name,
#         }
#         preprocessed_frames.append(preprocessed_frame)

#     logger.info(
#         f"Round {round_idx}: {error_frame_count}/{total_frames} frames skipped, {len(preprocessed_frames)} frames preprocessed."
#     )
#     return preprocessed_frames

def _build_preprocessed_frame(
    round_data: dict,
    frame_data: dict,
    players_data: list[dict],
    bomb_data: dict,
    tactic: str,
    map_name: str,
) -> dict:
    """Assemble the final preprocessed frame dictionary."""
    return {
        "round_data": round_data,
        "frame_data": frame_data,
        "players_data": players_data,
        "bomb_data": bomb_data,
        "tactic": tactic,
        "map_name": map_name,
    }

def _extract_round_data(round_obj: dict) -> dict:
    """Extract round-level metadata fields."""
    return {key: round_obj[key] for key in KEYS_ROUND_LEVEL}


def _adjust_frame_seconds(frame: dict, bomb_event_data: dict) -> float:
    """
    Return adjusted frame timestamp accounting for bomb plant timing.
    
    When a bomb has been planted and the current tick is at or past the plant
    tick, the plant duration is added to the raw frame seconds.
    """
    bomb_tick = bomb_event_data["bombTick"]
    if bomb_tick is not None and frame["tick"] >= bomb_tick:
        return frame["seconds"] + bomb_event_data["bombSeconds"]
    return frame["seconds"]

def _extract_frame_data(frame: dict, bomb_event_data: dict) -> dict:
    """Extract and time-adjust frame-level metadata."""
    frame_data = {key: frame[key] for key in KEYS_FRAME_LEVEL}
    frame_data["seconds"] = _adjust_frame_seconds(frame, bomb_event_data)
    return frame_data

def map_weapon_to_id(weapon_name: str, logger=None) -> int:
    """Map weapon name string to numeric ID."""
    if weapon_name not in WEAPON_ID_MAPPING:
        if logger:
            logger.warning(f"Unknown weapon: {weapon_name}")
        return -1
    return WEAPON_ID_MAPPING[weapon_name]

def _extract_player_data(player: dict, logger=None) -> dict:
    """Extract and normalise a single player's attributes."""
    player_data = {key: player[key] for key in KEYS_PLAYER_LEVEL}
    player_data["activeWeapon"] = map_weapon_to_id(
        player_data["activeWeapon"], logger=logger
    )
    return player_data

def _extract_team_data(
    dm: DataManager,
    frame: dict,
    logger=None,
) -> list[dict]:
    """
    Extract and sort all T-side players for a frame.
    
    Players are sorted by their mapped index to ensure a consistent ordering
    across frames, which is required for downstream graph construction.
    """
    players = frame["t"]["players"]
    sorted_players = sorted(
        players,
        key=lambda p: dm.get_player_idx_mapped(p["name"], "t", frame),
    )
    return [_extract_player_data(p, logger=logger) for p in sorted_players]

def preprocess_round(
    dm: DataManager,
    round_idx: int,
    frame_tactic_map: Optional[dict[str, str]] = None,
    logger=None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """
    Preprocess a single round: extract raw frame and player data.

    Returns a list of preprocessed frame dicts, one per valid frame. Each
    dict contains round_data, frame_data, players_data, bomb_data, tactic,
    and map_name.
    """
    round_obj = dm.get_game_round(round_idx)
    map_name = dm.get_map_name()
    round_data = _extract_round_data(round_obj)
    bomb_event_data = stats.process_bomb_data(round_obj)

    frames = dm._get_frames(round_idx)
    logger.info(f"Preprocessing round {round_idx} with {len(frames)} frames.")

    preprocessed_frames = []
    skipped = 0

    for frame_idx, frame in enumerate(frames):
        valid_frame, err_text = stats.check_frame_validity(frame)
        if not valid_frame:
            logger.warning(f"Frame {frame_idx} skipped: {err_text}")
            skipped += 1
            if strict:
                raise ValueError(f"Invalid frame {frame_idx}: {err_text}")
            continue

        tactic = frame_tactic_map.get(str(frame_idx), "unknown") if frame_tactic_map else "unknown"
        frame_data = _extract_frame_data(frame, bomb_event_data)
        players_data = _extract_team_data(dm, frame, logger=logger)
        bomb_data = dm.get_bomb_info(round_idx, frame_idx)

        preprocessed_frames.append(
            _build_preprocessed_frame(
                round_data, frame_data, players_data, bomb_data, tactic, map_name
            )
        )

    logger.info(
        f"Round {round_idx}: {skipped}/{len(frames)} frames skipped, "
        f"{len(preprocessed_frames)} frames preprocessed."
    )
    return preprocessed_frames

def process_single_file(args, logger):
    """
    Process a single demo file.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
    """
    logger.info(f"Starting preprocessing of {args.demo_path}")
    preprocessor = DataPreprocessor(Path(args.demo_path), logger=logger, strict=args.strict)
    
    preprocessed_rounds = preprocessor.preprocess_demo(
        tactic_labels_dir=args.tactic_labels_dir
    )
    
    # Save preprocessed data
    map_name = preprocessor.get_map_name()
    demo_name = Path(args.demo_path).stem
    output_folder = Path(args.output_dir) / map_name
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / demo_name
    
    with open(output_file, "wb") as f:
        pickle.dump(preprocessed_rounds, f)
    
    # Print summary
    total_frames = sum(len(frames) for frames in preprocessed_rounds.values())
    logger.info(f"Complete: {len(preprocessed_rounds)} rounds, {total_frames} total frames preprocessed")
    logger.info(f"Match ID: {preprocessor.get_match_id()}")
    logger.info(f"Map: {map_name}")
    logger.info(f"Saved to: {output_file}")

def load_tactic_labels(
    tactic_labels_dir: str,
    map_name: str,
    match_id: str,
    round_idx: int,
    logger=None,
) -> dict[str, str]:
    """
    Load per-frame tactic labels for a specific round.
    
    Returns dict mapping frame index (as string) to tactic label.
    """
    project_root = find_project_root()
    round_label_path = (
        project_root / Path(tactic_labels_dir) / map_name / match_id / f"{match_id}_{round_idx + 1}.json"
    )
    print(f"Loading tactic labels from {round_label_path}...")
    
    if round_label_path.exists():
        with open(round_label_path) as f:
            return json.load(f)
    else:
        if logger:
            logger.warning(
                f"No tactic labels found for round {round_idx + 1}. Defaulting to 'unknown'."
            )
        return {}
    
# def process_demo_batch(demo_path, queue=None, key=None, output_dir="data/preprocessed", tactic_labels_dir="data/tactic_labels", strict=False):
#     """
#     Process a single demo file in batch mode.
    
#     This must be a module-level function for multiprocessing compatibility.
#     """
#     demo_uuid = Path(demo_path).stem
    
#     try:
#         preprocessor = DataPreprocessor(Path(demo_path), strict=strict)
#         map_name = preprocessor.get_map_name()
#         preprocessed_rounds = preprocessor.preprocess_demo(tactic_labels_dir=tactic_labels_dir)
        
#         total_frames = sum(len(frames) for frames in preprocessed_rounds.values())
        
#         # Save
#         output_folder = Path(output_dir) / map_name
#         output_folder.mkdir(parents=True, exist_ok=True)
#         output_file = output_folder / f"{demo_uuid}.pkl"

#         with open(output_file, "wb") as f:
#             pickle.dump(preprocessed_rounds, f)

#         if queue and key:
#             queue.put((key, total_frames))
        
#         return {"status": "success", "frames": total_frames, "demo": demo_path}
#     except Exception as e:
#         if queue and key:
#             queue.put((key, 0))
#         return {"status": "failed", "error": str(e), "demo": demo_path}















# def process_batch(args, logger):
#     """
#     Process multiple demo files in batch mode.
    
#     Args:
#         args: Parsed command line arguments
#         logger: Logger instance
#     """
#     # Load environment variables
#     batch_size = args.processes or int(os.environ.get("CREATE_GRAPHS_PROCESSES_COUNT", 4))
#     demo_filenames_path = os.environ.get("DUST2_DEMOS_FILENAMES_PATH")
#     create_graphs_filenames = os.environ.get("CREATE_GRAPHS_FILENAMES_PATH")
#     create_graphs_demo_dir = os.environ.get("CREATE_GRAPHS_DEMO_DIR")
#     tactic_labels_dir = args.tactic_labels_dir
#     output_dir = args.output_dir
    
#     # Validate configuration
#     if not create_graphs_demo_dir:
#         raise ValueError("Environment variable CREATE_GRAPHS_DEMO_DIR is not set.")
#     if not os.path.exists(create_graphs_demo_dir):
#         raise ValueError(f"Demo directory {create_graphs_demo_dir} does not exist.")
    
#     filenames_path = create_graphs_filenames or demo_filenames_path
#     if not filenames_path:
#         raise ValueError("Neither CREATE_GRAPHS_FILENAMES_PATH nor DUST2_DEMOS_FILENAMES_PATH is set.")
#     if not os.path.exists(filenames_path):
#         raise ValueError(f"Demo filenames file {filenames_path} does not exist.")
    
#     # Load demo list
#     with open(filenames_path) as f:
#         filtered_demos = json.load(f)
    
#     logger.info(f"Found {len(filtered_demos)} demos in filenames list")
    
#     # Get actual demo paths
#     demo_pathnames = [
#         create_graphs_demo_dir + (demo_filename if isinstance(demo_filename, str) else demo_filename.get("filename", ""))
#         for demo_filename in filtered_demos
#         if os.path.exists(create_graphs_demo_dir + (demo_filename if isinstance(demo_filename, str) else demo_filename.get("filename", "")))
#     ]
    
#     logger.info(f"Found {len(demo_pathnames)} demo files in '{create_graphs_demo_dir}'")
    
#     # Filter out already processed unless reprocessing
#     if not args.reprocess:
#         original_count = len(demo_pathnames)
#         filtered_demos_list = []
#         for demo in demo_pathnames:
#             try:
#                 temp_dm = DataManager(Path(demo), do_validate=args.strict)
#                 map_name = temp_dm.get_map_name()
#                 demo_uuid = Path(demo).stem
#                 if not (Path(output_dir) / map_name / f"{demo_uuid}.pkl").exists():
#                     filtered_demos_list.append(demo)
#             except Exception as e:
#                 logger.warning(f"Could not check status of {demo}: {e}")
#                 filtered_demos_list.append(demo)
#         demo_pathnames = filtered_demos_list
#         skipped = original_count - len(demo_pathnames)
#         if skipped > 0:
#             logger.info(f"Skipping {skipped} already preprocessed demos (use --reprocess to override)")
    
#     if not demo_pathnames:
#         logger.info("No demos to process!")
#         return
    
#     logger.info(f"Processing {len(demo_pathnames)}/{len(filtered_demos)} demo files...")
    
#     # Process
#     if args.sync:
#         logger.info("Running in synchronous mode (single process)")
#         for demo in demo_pathnames:
#             process_demo_batch(
#                 demo,
#                 queue=None,
#                 key=None,
#                 output_dir=output_dir,
#                 tactic_labels_dir=tactic_labels_dir,
#                 strict=args.strict
#             )
#     else:
#         logger.info(f"Running in parallel mode ({batch_size} processes)")
        
#         # Calculate total frames
#         total_map = {}
#         for demo in demo_pathnames:
#             try:
#                 dm = DataManager(Path(demo), do_validate=args.strict)
#                 total_map[demo] = len(dm.get_all_frames())
#             except Exception as e:
#                 logger.warning(f"Could not estimate frames for {demo}: {e}")
#                 total_map[demo] = 0
        
#         # Batch processing with multiprocessing
#         manager = mp.Manager()
#         queue = manager.Queue()
#         monitor = mp.Process(target=progress_monitor, args=(queue, total_map))
#         monitor.start()
        
#         with ProcessPoolExecutor(max_workers=batch_size) as executor:
#             futures = [
#                 executor.submit(
#                     process_demo_batch,
#                     demo,
#                     queue=queue,
#                     key=demo,
#                     output_dir=output_dir,
#                     tactic_labels_dir=tactic_labels_dir,
#                     strict=args.strict
#                 )
#                 for demo in demo_pathnames
#             ]
            
#             for _ in as_completed(futures):
#                 pass
        
#         queue.put(None)
#         monitor.join()
    
#     logger.info("Batch preprocessing complete")


class DataPreprocessor:
    """
    Preprocesses CS:GO/CS2 demo files into raw frame and player data.
    """

    def __init__(self, demo_path: Path, logger=None, strict: bool = False):
        """
        Initialize preprocessor with a demo file.
        
        Args:
            demo_path: Path to the demo file
            logger: Logger instance
            strict: If True, raise errors on invalid frames instead of skipping
        """
        self.demo_path = Path(demo_path)
        self.logger = logger or logging.getLogger(__name__)
        self.strict = strict
        self.dm = DataManager(self.demo_path, do_validate=strict, logger=self.logger)

    def preprocess_demo(
        self,
        tactic_labels_dir: Optional[str] = None,
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Preprocess entire demo file.
        
        Returns dict mapping round index to list of preprocessed frames.
        """
        preprocessed_rounds = {}

        for round_idx in range(self.dm.get_round_count()):
            # Swap player mapping at half-time (MR15 setting)
            if round_idx == 15:
                self.dm.swap_player_mapping()

            # Load tactic labels if available
            frame_tactic_map = {}
            if tactic_labels_dir:
                frame_tactic_map = load_tactic_labels(
                    tactic_labels_dir,
                    self.dm.get_map_name(),
                    self.dm.get_match_id(),
                    round_idx,
                    logger=self.logger,
                )

            # Preprocess round
            preprocessed_frames = preprocess_round(
                self.dm,
                round_idx,
                frame_tactic_map=frame_tactic_map,
                logger=self.logger,
                strict=self.strict,
            )

            preprocessed_rounds[round_idx] = preprocessed_frames

        self.logger.info(
            f"SUCCESSFULLY COMPLETED: {self.dm.get_match_id()} demo preprocessed."
        )
        return preprocessed_rounds

    def get_match_id(self) -> str:
        return self.dm.get_match_id()

    def get_map_name(self) -> str:
        return self.dm.get_map_name()

    def get_round_count(self) -> int:
        return self.dm.get_round_count()


