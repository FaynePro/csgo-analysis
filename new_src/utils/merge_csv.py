import csv
import os
from pathlib import Path

from dotenv import load_dotenv

from graphs_to_csv import CSV_HEADERS

load_dotenv()

def main():
  graphs_folder = Path(os.environ.get("GRAPHS_OUTPUT_DIR"))
  output_name = "output.csv"
  try:
    with open(output_name, "w", newline="", encoding="utf-8") as f_out:
      print(f"Writing output to {output_name} from {graphs_folder}")
      writer = csv.writer(f_out)
      writer.writerow(CSV_HEADERS)
      csv_files = list(graphs_folder.rglob("*.csv")) # for len
      len_csv_files = len(csv_files)
      for idx, csv_file in enumerate(csv_files):
        file_name = csv_file.name
        demo_name = csv_file.parent.name
        round_idx = csv_file.stem.rpartition("-")[2]
        print(f"Loading demo round file: {file_name} ({idx+1}/{len_csv_files})")
        with open(csv_file, "r") as f:
          csvreader = csv.reader(f)
          count_frames = 0
          next(csvreader, None)  # skip the headers
          for frame in csvreader:
            writer.writerow(frame)
            count_frames += 1
          print(f"Written {count_frames} frames to file.")
    return
  except Exception as e:
    print(f"Failed: {e}")


if __name__ == "__main__":
  main()
