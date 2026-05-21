import csv
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OUTPUT_DIR = os.environ.get("CSV_OUTPUT_DIR")

KEYS_ROUND_LEVEL = (
  "tick",
  "seconds",
  "bombPlanted",
  "tFreezeTimeEndEqVal",
  "tRoundSpendMoney",
  "roundNum",
  "isWarmup",
  "winningSide",
  "losingTeam",
  "tRoundStartEqVal",
  "tactic_used"   # WARNING: older graph files have strategy_used, newer graph files have tactic_used
)

KEYS_PLAYER_LEVEL = (
  "x",
  "y",
  "z",
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
  "hasHelmet",
  "hasDefuse",
  "hasBomb",
  "areaId"
)

KEYS_EDGE_LEVEL = (
  "dist"
  # mean/min/max(dist) to bomb
  # mean/min/maxmin(dist) to siteA
  # mean/min/max(dist) to site B
  # sum(dist) all players (and bomb?)
)

# generate CSV headers
CSV_HEADERS = ("demoName", "roundIdx") + \
              KEYS_ROUND_LEVEL + \
              tuple([f"t{i}_{item}" for i in range(5) for item in KEYS_PLAYER_LEVEL]) + \
              ("dist01", "dist02", "dist03", "dist04", "dist06", "dist07", "dist08",
              "dist10", "dist12", "dist13", "dist14", "dist16", "dist17", "dist18",
              "dist20", "dist21", "dist23", "dist24", "dist26", "dist27", "dist28",
              "dist30", "dist31", "dist32", "dist34", "dist36", "dist37", "dist38",
              "dist40", "dist41", "dist42", "dist43", "dist46", "dist47", "dist48",
              "dist60", "dist61", "dist62", "dist63", "dist64", "dist67", "dist68")

# TODO
# create sequences and dependent variables:
#   - RoundWin
#   - change in hp/damage in next 5 seconds
#   - Kill in next 5 seconds
#   - distance from bomb to closest bombsite
#   - minimum distance from player to bombsite
# compare with with soccer possession value. Kill/Plant in next 5 second? Bomb distance to plant


def parse_graph_data(graph_data):
  # Handle missing keys gracefully - use None or 0 as default if key doesn't exist
  result = []
  for key in KEYS_ROUND_LEVEL:
    if key in graph_data:
      result.append(graph_data[key])
    else:
      # Default values for missing keys
      result.append(None)
  return result


def parse_node_data(node_data):
  # bug in create_graphs.py: players are ids 0,1,2,3,4 but bomb and bombsites are 6,7,8
  # players are index 0-4, ignore bomb and bombsites, they have no node attributes
  # iterate via range because order of index must remain constant
  result = []
  for i in range(5):
    for key in KEYS_PLAYER_LEVEL:
      if i in node_data and key in node_data[i]:
        result.append(node_data[i][key])
      else:
        result.append(None)
  return result


def parse_edges_data(edges_data):
  # make sure we always have same order
  edges_data.sort(key=lambda x: (x[0], x))
  result = []
  for edge in edges_data:
    if len(edge) > 2 and 'dist' in edge[2]:
      result.append(edge[2]['dist'])
    else:
      result.append(None)
  return result


def main():
  graphs_folder = Path(os.environ.get("GRAPHS_OUTPUT_DIR"))
  output_name = f"{OUTPUT_DIR}/output.csv"
  try:
    with open(output_name, "w", newline="", encoding="utf-8") as f_out:
      print(f"Writing output to {output_name} from {graphs_folder}")
      writer = csv.writer(f_out)
      writer.writerow(CSV_HEADERS)
      pkl_files = list(graphs_folder.rglob("*.pkl"))
      len_pkl_files = len(pkl_files)
      
      if len_pkl_files == 0:
        print(f"Warning: No .pkl files found in {graphs_folder}")
        return
      
      frames_written = 0
      for idx, pkl_file in enumerate(pkl_files):
        file_name = pkl_file.name
        demo_name = pkl_file.parent.name
        round_idx = pkl_file.stem.rpartition("-")[2]
        print(f"Loading demo round file: {file_name} ({idx+1}/{len_pkl_files})")
        
        try:
          with open(pkl_file, "rb") as f:
            frames = pickle.load(f)
            for frame in frames:
              try:
                row = [demo_name, round_idx] + \
                      parse_graph_data(frame["graph_data"]) + \
                      parse_node_data(frame["nodes_data"]) + \
                      parse_edges_data(frame["edges_data"])
                writer.writerow(row)
                frames_written += 1
              except KeyError as e:
                print(f"  Warning: Skipped frame due to missing key {e}")
                continue
            print(f"  Written {len(frames)} frames from {file_name}")
        except Exception as e:
          print(f"  Error loading {pkl_file}: {e}")
          continue
      
      print(f"✅ Successfully wrote {frames_written} total frames to {output_name}")
      return
  except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()


def main_onlyprint():
  graphs_folder = Path(os.environ.get("GRAPHS_OUTPUT_DIR"))
  output_name = f"{OUTPUT_DIR}/output.csv"
  print(f"Writing output to {output_name} from {graphs_folder}")
  pkl_files = list(graphs_folder.rglob("*.pkl")) # for len
  len_pkl_files = len(pkl_files)
  for idx, pkl_file in enumerate(pkl_files):
    file_name = pkl_file.name
    demo_name = pkl_file.parent.name
    round_idx = pkl_file.stem.rpartition("-")[2]
    print(f"Loading demo round file: {file_name} ({idx+1}/{len_pkl_files})")
    with open(pkl_file, "rb") as f:
      frames = pickle.load(f)
      for frame in frames:
        pass #todo add print
      print(f"Written {len(frames)} to file.")
  return

def run():
  main()
  return "Extraction complete. Check output.csv for results."

if __name__ == "__main__":
  main()
