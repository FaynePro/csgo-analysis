import argparse
import csv
import functools
import json
import logging
import os
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from awpy.analytics.nav import area_distance, find_closest_area
from awpy.data import AREA_DIST_MATRIX, NAV
from dotenv import load_dotenv
from tqdm import tqdm

import stats
from datamodel.data_manager import DataManager
from graphs_to_csv import parse_graph_data, parse_node_data, parse_edges_data, CSV_HEADERS
from utils.discord_webhook import send_progress_embed
from utils.download_demo_from_repo import get_demo_files_from_list
from utils.logging_config import get_logger

load_dotenv()

# Remove any default root handlers (they print to console)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set root level to WARNING or higher (to suppress DEBUG logs)
# TODO: change to environment variable
logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("discord").setLevel(logging.CRITICAL)
logging.getLogger("discord.webhook.async_").setLevel(logging.CRITICAL)

KEYS_ROUND_LEVEL = (
    "roundNum",
    "isWarmup",
    "winningSide",
    "losingTeam",
    "tFreezeTimeEndEqVal",
    "tRoundStartEqVal",
    "tRoundSpendMoney")

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
KEYS_PER_NODE = KEYS_PLAYER_LEVEL + (
    "areaId",
    "nodeType",
)  # areaId and nodeType are added during processing

# DGL library requires int as node ids
# instead of randomly converting later, ensure that it is always 6,7,8, because
# players are 1-5
# TODO: players are actually 0-4, make consistent!
BOMB_NODE_INDEX = 6
BOMBSITE_A_NODE_INDEX = 7
BOMBSITE_B_NODE_INDEX = 8

# PyTorch can only handle numeric tensors
NODE_TYPE_PLAYER_INDEX = (
    1000  # players are more like bomb than targets (based on attributes)
)
NODE_TYPE_BOMB_INDEX = 900
NODE_TYPE_TARGET_INDEX = 1
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


def process_round(
    dm: DataManager,
    round_idx: int,
    frame_tactic_map: dict[str, str] = None,
    queue=None,
    key=None,
    logger=None,
    strict=False,
) -> list[list[Any]]:
    round = dm.get_game_round(round_idx)
    map_name = dm.get_map_name()

    # all variables on the round level --> graph data
    round_data = {key: round[key] for key in KEYS_ROUND_LEVEL}

    frames = dm._get_frames(round_idx)

    logger.info("Processing round %d with %d frames." % (round_idx, len(frames)))

    # store crucial bomb events for later analysis and estimating correct round ingame seconds.
    bomb_event_data = stats.process_bomb_data(round)

    # iterate and process each frame
    graphs = []
    error_frame_count = 0
    total_frames = len(frames)
    # Remove local tqdm bar, progress will be reported via queue
    for frame_idx, frame in enumerate(frames):
        # check validity of frame
        valid_frame, err_text = stats.check_frame_validity(frame)
        if not valid_frame:
            logger.warning(f"Frame {frame_idx} skipped: {err_text}")
            if strict:
                raise ValueError(f"Invalid frame {frame_idx}: {err_text}")
            if queue and key:
                queue.put((key, 1))
            continue

        # tactic label for this frame
        tactic = (
            frame_tactic_map.get(str(frame_idx), "unknown")
            if frame_tactic_map
            else "unknown"
        )

        # all variables on the frame level are added to the graph level data.
        graph_data = {key: frame[key] for key in KEYS_FRAME_LEVEL} | round_data
        graph_data["tactic_used"] = tactic

        # include estimated seconds from bomb data for each frame
        if (
            bomb_event_data["bombTick"] is not None
            and frame["tick"] >= bomb_event_data["bombTick"]
        ):
            graph_data["seconds"] = frame["seconds"] + bomb_event_data["bombSeconds"]
        else:
            graph_data["seconds"] = frame["seconds"]

        # all variables on the team and player level for the T side
        team = frame["t"]

        ### Create Node Data
        # iterate through all players, but keep them in same order every iteration
        nodes_data = {}
        edges_data = []
        for player_idx, player in enumerate(
            sorted(
                team["players"],
                key=lambda p: dm.get_player_idx_mapped(p["name"], "t", frame),
            )
        ):
            node_data = {key: player[key] for key in KEYS_PLAYER_LEVEL}
            node_data["areaId"] = find_closest_area(
                map_name, point=[node_data[key] for key in ("x", "y", "z")], flat=False
            )["areaId"]
            node_data["nodeType"] = NODE_TYPE_PLAYER_INDEX
            node_data["activeWeapon"] = map_weapon_to_id(
                node_data["activeWeapon"], logger=logger
            )
            nodes_data[player_idx] = node_data

        # add bomb node
        nodes_data[BOMB_NODE_INDEX] = dm.get_bomb_info(round_idx, frame_idx)
        nodes_data[BOMB_NODE_INDEX]["areaId"] = find_closest_area(
            map_name,
            point=[nodes_data[BOMB_NODE_INDEX][key] for key in ("x", "y", "z")],
            flat=False,
        )["areaId"]
        nodes_data[BOMB_NODE_INDEX]["nodeType"] = NODE_TYPE_BOMB_INDEX

        ### Create Edge Data
        # computes distances to bombsite
        try:
            distance_A, distance_B = distance_bombsites(dm, nodes_data, logger=logger)
        except ValueError as exc:
            logger.warning(
                f"Frame {frame_idx} in round {round_idx} skipped due to: {exc}"
            )
            error_frame_count += 1
            if strict:
                raise
            if queue and key:
                queue.put((key, 1))
            continue  # skip errors
        for k in nodes_data:
            edges_data.append((k, BOMBSITE_A_NODE_INDEX, {"dist": distance_A[k]}))
            edges_data.append((k, BOMBSITE_B_NODE_INDEX, {"dist": distance_B[k]}))

        # compute distances pairwise
        for node_a in reversed(nodes_data.keys()):
            for node_b in reversed(nodes_data.keys()):
                # ignore self loops
                if node_a == node_b:
                    continue
                edges_data.append(
                    (
                        node_a,
                        node_b,
                        {
                            "dist": _distance_internal(
                                map_name,
                                nodes_data[node_a]["areaId"],
                                nodes_data[node_b]["areaId"],
                                logger=logger,
                            )
                        },
                    )
                )

        # add target site nodes after all distance calcuations, so we can always just take the entire dict as input
        nodes_data[BOMBSITE_A_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}
        nodes_data[BOMBSITE_B_NODE_INDEX] = {"nodeType": NODE_TYPE_TARGET_INDEX}

        # fill up all keys with empty values, because all nodes need same attributes for DGL
        for k in nodes_data:
            nodes_data[k] = fill_keys(nodes_data[k])  # merging dicts creates a copy

        # store data in convenient dict
        graph = {
            "graph_data": graph_data,
            "nodes_data": nodes_data,
            "edges_data": edges_data,
        }
        graphs.append(graph)
        if queue and key:
            queue.put((key, 1))
    logger.info(
        f"Round {round_idx}: {error_frame_count}/{total_frames} frames skipped."
    )
    return graphs


def map_weapon_to_id(weaponName: str, logger=None) -> int:
    if weaponName not in WEAPON_ID_MAPPING:
        # Optional: log unknown weapons for later inspection
        logging.getLogger("weapon_mapping").warning(f"Unknown weapon: {weaponName}")
        return -1  # or a reserved ID like -1
    return WEAPON_ID_MAPPING[weaponName]


def fill_keys(target: dict):
    empty_dict = {key: 0 for key in KEYS_PER_NODE}  # None does not work as tensor
    return empty_dict | target  # right dict takes precedence


def distance_bombsites(dm: DataManager, nodes: dict, logger=None):
    map_name = dm.get_map_name()

    if map_name not in NAV:
        raise ValueError("Map not found.")

    # find shortest distances to both bombsites:
    closest_distances_A = {key: float("Inf") for key in nodes}
    closest_distances_B = {key: float("Inf") for key in nodes}

    ## Todo: find bombsite *plantable* area  with minimum distance from bomb
    for map_area_id in NAV[map_name]:
        map_area = NAV[map_name][map_area_id]
        if map_area["areaName"].startswith("BombsiteA"):
            for key, _value in nodes.items():
                target_area = nodes[key]["areaId"]
                current_bombsite_dist = _distance_internal(
                    map_name, map_area_id, target_area
                )
                # Set closest distance
                if current_bombsite_dist < closest_distances_A[key]:
                    closest_distances_A[key] = current_bombsite_dist

        elif map_area["areaName"].startswith("BombsiteB"):
            for key, _value in nodes.items():
                target_area = nodes[key]["areaId"]
                current_bombsite_dist = _distance_internal(
                    map_name, map_area_id, target_area
                )
                # Set closest distance
                if current_bombsite_dist < closest_distances_B[key]:
                    closest_distances_B[key] = current_bombsite_dist

    # estimate the distances for nodes that are not reachable by neighbors
    for dist_dict in [closest_distances_A, closest_distances_B]:
        keys = list(dist_dict.keys())
        for i, key in enumerate(keys):
            if dist_dict[key] == float("Inf"):
                # Try to estimate using neighbors
                prev_dist = None
                next_dist = None

                # look backwards
                for j in range(i - 1, -1, -1):
                    if dist_dict[keys[j]] != float("Inf"):
                        prev_dist = dist_dict[keys[j]]
                        break

                # look forward
                for j in range(i + 1, len(keys)):
                    if dist_dict[keys[j]] != float("Inf"):
                        next_dist = dist_dict[keys[j]]
                        break

                if prev_dist is not None and next_dist is not None:
                    dist_dict[key] = (prev_dist + next_dist) / 2
                elif prev_dist is not None:
                    dist_dict[key] = prev_dist
                elif next_dist is not None:
                    dist_dict[key] = next_dist
                else:
                    raise ValueError(
                        f"Could not estimate closest bombsite distances for node '{key}'."
                    )

    # collate to tuple and return
    return closest_distances_A, closest_distances_B


def _distance_internal(map_name, area_a, area_b, logger=None):
    # Use Area Distance Matrix if available, since it is faster
    # distance matrix uses strings as key
    area_a_str = str(area_a)
    area_b_str = str(area_b)
    if (
        map_name in AREA_DIST_MATRIX
        and area_a_str in AREA_DIST_MATRIX[map_name]
        and area_b_str in AREA_DIST_MATRIX[map_name][area_a_str]
    ):
        current_bombsite_dist = AREA_DIST_MATRIX[map_name][area_a_str][area_b_str][
            "geodesic"
        ]
    # Else: calculate distance from pairwise iteration over all areas in map
    else:
        if logger and len(AREA_DIST_MATRIX) > 0:
            logger.warning(
                "Area matrix exists but does not contain areaid: %d" % area_a
            )
        geodesic_path = area_distance(
            map_name=map_name, area_a=area_a, area_b=area_b, dist_type="geodesic"
        )
        current_bombsite_dist = geodesic_path["distance"]

    return current_bombsite_dist


def process_single_demo(
    demo_path,
    queue=None,
    key=None,
    send_dc_webhooks=False,
    rewrite_graphed_rounds=False,
    strict=False,
    tactic_labels_dir: str="data/tactic_labels",
    create_graphs_output_dir: str="data/graphs",
    output_type="pickle"
):
    # logger
    demo_uuid = Path(demo_path).stem
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = Path(create_graphs_output_dir) / demo_uuid / "logs" / f"{timestamp}.log"
    logger = get_logger(
        log_path, name=f"create_graphs_logger_{demo_uuid}", level=logging.DEBUG
    )

    dm = DataManager(Path(demo_path), do_validate=strict, logger=logger)
    output_folder = Path(create_graphs_output_dir) / demo_uuid
    if output_type == "pickle":
        output_filename_template = str(output_folder / "graph-rounds-%d.pkl")
    elif output_type == "csv":
        output_filename_template = str(output_folder / "graph-rounds-%d.csv")
    else:
        raise ValueError(f"Output type {output_type} is not supported.")
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Processing match id: %s with %d rounds."
        % (dm.get_match_id(), dm.get_round_count())
    )

    # tactic label directory for per-frame labeling
    tactic_dir = (
        Path(tactic_labels_dir)
        / dm.get_map_name()
        / dm.get_match_id()
    )

    start_time = time.time()
    total_frames = len(dm.get_all_frames())
    processed_frames = 0
    graphs_total = 0
    for round_idx in range(dm.get_round_count()):
        output_filename = output_filename_template % round_idx
        logger.info("Converting round %d to file %s." % (round_idx, output_filename))

        progress = round((processed_frames / total_frames) * 100, 2)
        eta = dm.get_estimated_finish(
            start_time=start_time, processed_frames=processed_frames
        )
        if send_dc_webhooks:
            send_progress_embed(
                progress=progress,
                roundsTotal=dm.get_round_count(),
                currentRound=round_idx,
                eta=eta,
                id=dm.get_match_id(),
                sendSilent=(
                    round_idx not in [0, dm.get_round_count() - 1]
                ),  # Send silent for first and last round
                logger=logger,
            )

        # we need to swap mappings, because player sides switch here.
        # WARNING: This only works if teams player in MR15 setting.
        if round_idx == 15:
            dm.swap_player_mapping()

        # Load per-frame tactic labels for this round
        round_label_path = tactic_dir / f"{dm.get_match_id()}_{round_idx + 1}.json"
        if round_label_path.exists():
            with open(round_label_path) as f:
                frame_tactic_map = json.load(f)
        else:
            logger.warning(
                f"No tactic labels found for round {round_idx + 1}. Defaulting to 'unknown'."
            )
            frame_tactic_map = {}

        # Skip if not rewriting and file exists
        if not rewrite_graphed_rounds and Path(output_filename).exists():

            logger.info(
                f"Skipping round {round_idx} with {len(dm._get_frames(round_idx))} frames: graph file already exists."
            )
            estimated_frames = len(dm._get_frames(round_idx))
            if queue and key:
                queue.put((key, estimated_frames))

            graphs_total += estimated_frames
            processed_frames += estimated_frames
            continue

        graphs = process_round(
            dm,
            round_idx,
            frame_tactic_map=frame_tactic_map,
            queue=queue,
            key=key,
            logger=logger,
            strict=strict,  # reuse flag for now
        )

        if output_type == "pickle":
            with open(output_filename, "wb") as f:
                pickle.dump(graphs, f)
        elif output_type == "csv":
            with open(output_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
                for graph in graphs:
                    writer.writerow(
                        [dm.get_match_id(), round_idx] +
                        parse_graph_data(graph["graph_data"]) +
                        parse_node_data(graph["nodes_data"]) +
                        parse_edges_data(graph["edges_data"]))
        else:
            raise ValueError(f"Output type {output_type} is not supported.")

        logger.info("%d graphs written to file." % len(graphs))
        graphs_total += len(graphs)
        processed_frames += len(graphs)

    logger.info("✅ SUCCESSFULLY COMPLETED: %d graphs written in total." % graphs_total)

    if processed_frames < total_frames:
        logger.warning(f"{total_frames - processed_frames} frames were skipped.")
    logger.info(f"Processed {processed_frames} / {total_frames} frames.")

    if send_dc_webhooks:
        send_progress_embed(
            progress=100,
            roundsTotal=dm.get_round_count(),
            currentRound=dm.get_round_count() - 1,
            eta=0,
            id=dm.get_match_id(),
            sendSilent=False,
            logger=logger,
        )


def progress_monitor(queue, total_map):
    pbars = {
        k: tqdm(total=v, desc=k, position=i, leave=True)
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


def get_env_variables():
    batch_size = int(os.environ.get("CREATE_GRAPHS_PROCESSES_COUNT", 10))
    demo_filenames_path = os.environ.get("DUST2_DEMOS_FILENAMES_PATH")
    create_graphs_filenames = os.environ.get("CREATE_GRAPHS_FILENAMES_PATH")
    create_graphs_demo_dir = os.environ.get("CREATE_GRAPHS_DEMO_DIR")
    output_type = os.environ.get("CREATE_GRAPHS_OUTPUT_TYPE")
    create_graphs_output_dir = os.environ.get("GRAPHS_OUTPUT_DIR")
    tactic_labels_dir = os.environ.get("LABELS_OUTPUT_DIR")

    if not create_graphs_demo_dir:
        raise ValueError("Environment variable CREATE_GRAPHS_DEMO_DIR is not set.")
    if not os.path.exists(create_graphs_demo_dir):
        raise ValueError(f"Demo directory {create_graphs_demo_dir} does not exist. Please check the path.")

    if not demo_filenames_path:
        raise ValueError("Environment variable DUST2_DEMOS_FILENAMES_PATH is not set.")
    if not os.path.exists(demo_filenames_path):
        raise ValueError(f"File list path {demo_filenames_path} does not exist. Please check the path.")

    if not create_graphs_filenames:
        raise ValueError("Environment variable CREATE_GRAPHS_FILENAMES_PATH is not set.")
    if not os.path.exists(create_graphs_filenames):
        raise ValueError(f"File list path {create_graphs_filenames} does not exist. Please check the path.")

    if not tactic_labels_dir:
        raise ValueError("Environment variable LABELS_OUTPUT_DIR is not set.")
    if not os.path.exists(tactic_labels_dir):
        raise ValueError(f"Graph output path {tactic_labels_dir} does not exist. Please check the path.")

    if not create_graphs_output_dir:
        raise ValueError("Environment variable GRAPHS_OUTPUT_DIR is not set.")
    if not os.path.exists(create_graphs_output_dir):
        raise ValueError(f"Graph output path {create_graphs_output_dir} does not exist. Please check the path.")

    if batch_size:
        print(f"Using {batch_size} processes for graph creation.")
    else:
        print("Environment variable CREATE_GRAPHS_PROCESSES_COUNT is not set. Using default of 1 process.")
        batch_size = 1

    if not output_type or not output_type in ("pickle", "csv"):
        print("Environment variable CREATE_GRAPHS_CSV_OUTPUT is not set to valid value ('pickle' or 'csv'). " \
              "Using default of 'pickle'.")
        output_type = "pickle"

    return (batch_size,
        demo_filenames_path,
        create_graphs_filenames,
        create_graphs_demo_dir,
        tactic_labels_dir,
        create_graphs_output_dir,
        output_type)


def main(send_dc_webhooks=False, rewrite_graphed_rounds=False, strict=False, sync=False):

    batch_size, demo_filenames_path, create_graphs_filenames, \
        create_graphs_demo_dir, tactic_labels_dir, create_graphs_output_dir, \
        output_type = get_env_variables()

    demo_filenames = get_demo_files_from_list(demo_filenames_path, compressed=False)

    print(f"Found {len(demo_filenames)} demo filenames in the demo filenames list.")

    with open(create_graphs_filenames) as f:
        filtered_demos = json.load(f)
    print(
        f"Found {len(filtered_demos)} demo filenames in the scheduled process file list."
    )

    demo_pathnames = [
        create_graphs_demo_dir + demo_filename
        for demo_filename in filtered_demos
        if os.path.exists(create_graphs_demo_dir + demo_filename)
    ]

    print(
        f"Found {len(demo_pathnames)} demo files in '{create_graphs_demo_dir}' directory."
    )
    print(f"Processing {len(demo_pathnames)}/{len(filtered_demos)} demo files...")

    # Calculate total frames per demo for progress bars
    total_map = {
        demo: len(DataManager(Path(demo), do_validate=strict).get_all_frames())
        for demo in demo_pathnames
    }

    if sync:
        for demo in demo_pathnames:
            process_single_demo(
                demo,
                None,
                None,
                send_dc_webhooks=send_dc_webhooks,
                rewrite_graphed_rounds=rewrite_graphed_rounds,
                strict=strict,
                tactic_labels_dir=tactic_labels_dir,
                create_graphs_output_dir=create_graphs_output_dir,
                output_type=output_type
            )
    else:
        # Create a multiprocessing queue for communication with the monitor
        manager = mp.Manager()
        queue = manager.Queue()
        # Launch the monitor process
        # Process terminates when "None" is retrieved from Queue
        monitor = mp.Process(target=progress_monitor, args=(queue, total_map))
        monitor.start()
        # Create a ProcessPoolExecutor with the desired number of workers
        with ProcessPoolExecutor(max_workers=batch_size) as executor:

            # Submit all tasks to the executor
            futures = [
                executor.submit(
                    functools.partial(
                        process_single_demo,
                        demo,
                        queue,
                        demo,
                        send_dc_webhooks=send_dc_webhooks,
                        rewrite_graphed_rounds=rewrite_graphed_rounds,
                        strict=strict,
                        tactic_labels_dir=tactic_labels_dir,
                        create_graphs_output_dir=create_graphs_output_dir,
                        output_type=output_type,
                    )
                )
                for demo in demo_pathnames
            ]

            # Process the completed tasks
            for future in as_completed(futures):
                try:
                    result = future.result()  # Get the task result (if returned)
                    print(f"Task completed with result: {result}")  # Optional: Process the result
                except Exception as e:
                    print(f"Task failed with error: {e}")  # Handle errors appropriately

        # Signal task completion to the queue and join the monitor
        queue.put(None)
        monitor.join()

def run():
    parser = argparse.ArgumentParser(description="Process CS:GO demo graphs.")
    # action is inverted, because absence of flag yes/no
    parser.add_argument("--no-dc-webhooks", action="store_false",
                        help="Disable Discord webhook progress updates (default: True)")
    parser.add_argument("--rewrite-graphed-rounds", action="store_true",
        help="Rewrite rounds even if graph files already exist (default: False)")
    parser.add_argument("--strict", action="store_true",
                        help="Raise an error if a frame is invalid (default: False)")
    parser.add_argument("--sync", action="store_true",
                        help="Run in synchronous mode without concurrent processing. (default: False)")
    args = parser.parse_args()

    print("Starting graph creation...")
    print(f"Arguments: {args}")

    main(send_dc_webhooks=not args.no_dc_webhooks,
        rewrite_graphed_rounds=args.rewrite_graphed_rounds,
        strict=args.strict,
        sync=args.sync)

    return "Finished processing graphs."

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process CS:GO demo graphs.")
    # action is inverted, because absence of flag yes/no
    parser.add_argument("--no-dc-webhooks", action="store_false",
                        help="Disable Discord webhook progress updates (default: True)")
    parser.add_argument("--rewrite-graphed-rounds", action="store_true",
        help="Rewrite rounds even if graph files already exist (default: False)")
    parser.add_argument("--strict", action="store_true",
                        help="Raise an error if a frame is invalid (default: False)")
    parser.add_argument("--sync", action="store_true",
                        help="Run in synchronous mode without concurrent processing. (default: False)")
    args = parser.parse_args()

    print("Starting graph creation...")
    print(f"Arguments: {args}")

    main(send_dc_webhooks=not args.no_dc_webhooks,
        rewrite_graphed_rounds=args.rewrite_graphed_rounds,
        strict=args.strict,
        sync=args.sync)