import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datamodel.data_manager import DataManager
from tqdm import tqdm
from preprocessing.functions.data_preprocessor import DataPreprocessor
from utils.project_root import find_project_root

def _save_preprocessed_demo(
    preprocessed_rounds: dict,
    output_dir: str,
    map_name: str,
    demo_uuid: str,
) -> Path:
    """Persist preprocessed round data to disk and return the output path."""
    output_path = _get_output_path(output_dir, map_name, demo_uuid)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(preprocessed_rounds, f)

    return output_path

def progress_monitor(queue, total_map):
    """
    Monitor progress from multiprocessing tasks.
    """
    pbars = {
        k: tqdm(total=v, desc=str(k), position=i, leave=True)  # convert k to str, as k is a windows path
        for i, (k, v) in enumerate(total_map.items())
    }
    finished = set()
    while len(finished) < len(pbars):
        task = queue.get()
        if task is None:
            break
        key, n = task
        if key in pbars:
            pbars[key].update(n)
            if pbars[key].n >= pbars[key].total:
                finished.add(key)
    for pbar in pbars.values():
        pbar.close()

def _get_output_path(output_dir: str, map_name: str, demo_uuid: str) -> Path:
    """Resolve the output file path for a preprocessed demo."""
    return find_project_root() / Path(output_dir) / map_name / f"{demo_uuid}.pkl"

def _load_batch_config(args) -> tuple[Path, Path]:
    """Load and validate environment config for batch processing."""

    project_root = find_project_root()

    # ---- Preprocessed demo directory ----
    demo_dir_env = os.environ.get(args.batch_files_path)
    if not demo_dir_env:
        raise ValueError(
            f"Environment variable '{args.batch_files_path}' is not set."
        )

    preprocessed_demo_dir = project_root / demo_dir_env
    if not preprocessed_demo_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed demo directory not found: {preprocessed_demo_dir}"
        )
    if not preprocessed_demo_dir.is_dir():
        raise ValueError(
            f"Expected a directory but got: {preprocessed_demo_dir}"
        )

    # ---- JSON file containing selected demo filenames ----
    demo_list_env = os.environ.get(args.batch_file_names)
    if not demo_list_env:
        raise ValueError(
            f"Environment variable '{args.batch_file_names}' is not set."
        )

    demo_list_json_path = project_root / demo_list_env
    if not demo_list_json_path.exists():
        raise FileNotFoundError(
            f"Demo list JSON file not found: {demo_list_json_path}"
        )
    if not demo_list_json_path.is_file():
        raise ValueError(
            f"Expected a file but got: {demo_list_json_path}"
        )

    return preprocessed_demo_dir, demo_list_json_path


def _resolve_demo_paths(preprocessed_demo_dir: Path, demo_list_json_path: Path) -> list[Path]:
    """Load demo filenames and return paths that exist on disk."""
    with open(demo_list_json_path) as f:
        entries = json.load(f)

    def to_filename(entry) -> str:
        return entry if isinstance(entry, str) else entry.get("filename", "")

    return [
        preprocessed_demo_dir / to_filename(entry)
        for entry in entries
        if os.path.exists(preprocessed_demo_dir / to_filename(entry))
    ]


def _filter_unprocessed(demo_paths: list[str], output_dir: str, strict: bool, logger) -> list[str]:
    """Return only demos that have not yet been preprocessed."""
    pending = []
    for demo in demo_paths:
        try:
            dm = DataManager(Path(demo), do_validate=strict)
            output_path = _get_output_path(output_dir, dm.get_map_name(), Path(demo).stem)
            if not output_path.exists():
                pending.append(demo)
        except Exception as e:
            logger.warning(f"Could not check status of {demo}: {e}")
            pending.append(demo)
    return pending


def _run_sync(demo_paths: list[str], output_dir: str, tactic_labels_dir: str, strict: bool, logger) -> None:
    """Process demos sequentially in a single process."""
    logger.info("Running in synchronous mode (single process)")
    for demo in demo_paths:
        process_demo_batch(output_dir=output_dir, tactic_labels_dir=tactic_labels_dir, demo_path=demo, strict=strict)


def _run_parallel(demo_paths: list[str], output_dir: str, tactic_labels_dir: str, strict: bool, batch_size: int, logger) -> None:
    """Process demos in parallel using a process pool."""
    logger.info(f"Running in parallel mode ({batch_size} processes)")

    total_map = {}
    for demo in demo_paths:
        try:
            dm = DataManager(Path(demo), do_validate=strict)
            total_map[demo] = len(dm.get_all_frames())
        except Exception as e:
            logger.warning(f"Could not estimate frames for {demo}: {e}")
            total_map[demo] = 0

    manager = mp.Manager()
    queue = manager.Queue()
    monitor = mp.Process(target=progress_monitor, args=(queue, total_map))
    monitor.start()

    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                process_demo_batch,
                output_dir=output_dir,
                tactic_labels_dir=tactic_labels_dir,
                demo_path=demo,
                queue=queue,
                key=demo,
                strict=strict,
            )
            for demo in demo_paths
        ]
        for _ in as_completed(futures):
            pass

    queue.put(None)
    monitor.join()


def _count_total_frames(preprocessed_rounds: dict) -> int:
    """Sum frames across all rounds."""
    return sum(len(frames) for frames in preprocessed_rounds.values())


def _notify_progress(queue, key, frame_count: int) -> None:
    """Send a progress update to the monitor queue, if one is provided."""
    if queue and key:
        queue.put((key, frame_count))


def process_batch(args, logger) -> None:
    """
    Process multiple demo files in batch mode.

    Loads demo list from environment config, filters already-processed
    demos unless --reprocess is set, then dispatches work either
    synchronously or in parallel.
    """
    batch_size = args.processes or int(os.environ.get("CREATE_GRAPHS_PROCESSES_COUNT", 4))
    preprocessed_demo_dir, demo_list_json_path = _load_batch_config(args)
    processed_demo_paths = _resolve_demo_paths(preprocessed_demo_dir, demo_list_json_path)
    logger.info(f"Found {len(processed_demo_paths)} demo files in '{preprocessed_demo_dir}'")
    logger.info(f"Using filenames path: {demo_list_json_path}")

    if not args.reprocess:
        original_count = len(processed_demo_paths)
        processed_demo_paths = _filter_unprocessed(processed_demo_paths, args.output_dir, args.strict, logger)
        skipped = original_count - len(processed_demo_paths)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already preprocessed demos (use --reprocess to override)")

    if not processed_demo_paths:
        logger.info("No demos to process!")
        return

    logger.info(f"Processing {len(processed_demo_paths)} demo files...")

    if args.sync:
        _run_sync(processed_demo_paths, args.output_dir, args.tactic_labels_dir, args.strict, logger)
    else:
        _run_parallel(processed_demo_paths, args.output_dir, args.tactic_labels_dir, args.strict, batch_size, logger)

    logger.info("Batch preprocessing complete")


def process_demo_batch(
    output_dir: str,
    tactic_labels_dir: str,
    demo_path: str,
    queue=None,
    key=None,
    
    strict: bool = False,
) -> dict:
    """
    Process a single demo file in batch mode.

    Must be a module-level function for multiprocessing compatibility.

    Returns a result dict with keys: status, demo, and either
    frames (on success) or error (on failure).
    """
    demo_uuid = Path(demo_path).stem

    try:
        preprocessor = DataPreprocessor(Path(demo_path), strict=strict)
        map_name = preprocessor.get_map_name()
        preprocessed_rounds = preprocessor.preprocess_demo(tactic_labels_dir=tactic_labels_dir)

        total_frames = _count_total_frames(preprocessed_rounds)
        _save_preprocessed_demo(preprocessed_rounds, output_dir, map_name, demo_uuid)
        _notify_progress(queue, key, total_frames)

        return {"status": "success", "frames": total_frames, "demo": demo_path}

    except Exception as e:
        _notify_progress(queue, key, 0)
        return {"status": "failed", "error": str(e), "demo": demo_path}
    
    