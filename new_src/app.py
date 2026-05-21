import os
import sys
import json
import time
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).parent
PARSER_DIR = BASE_DIR / "parser"
SPLITTER_DIR = BASE_DIR / "splitter"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

LOG_DIR = OUTPUT_DIR / "logs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, LOG_DIR, MODEL_OUTPUT_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)


# Helper files
def discover_modules(folder: Path):
    modules = []

    for file in folder.glob("*.py"):
        if file.name.startswith("__"):
            continue

        modules.append(file.stem)

    return sorted(modules)



def run_subprocess(script_path: Path):
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []

    for line in process.stdout:
        output_lines.append(line)

    process.wait()

    return process.returncode, "".join(output_lines)



def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)

    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(path.parent.resolve()))

    spec.loader.exec_module(module)

    return module



def execute_module(module_path: Path, splitter_output=None):
    # Small note, this will run the run() function if there is one
    text = module_path.read_text(encoding="utf-8", errors="ignore").lower()

    multiprocessing_tokens = [
        "multiprocessing",
        "processpoolexecutor",
        "pool(",
        "concurrent.futures",
    ]

    needs_subprocess = any(t in text for t in multiprocessing_tokens)

    if needs_subprocess:
        return run_subprocess(module_path)

    try:
        module = load_module(module_path)

        if hasattr(module, "run"):
            if splitter_output is not None:
                result = module.run(splitter_output)
            else:
                result = module.run()
            return 0, result

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, type):
                if hasattr(attr, "run"):
                    instance = attr()
                    if splitter_output is not None:
                        result = instance.run(splitter_output)
                    else:
                        result = instance.run()
                    return 0, result

        return 1, "No runnable module found"

    except Exception as e:
        return 1, str(e)



def save_log(name, output):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = LOG_DIR / f"{name}_{timestamp}.log"

    with open(path, "w") as f:
        f.write(output)

    return path

def extract_metrics(model_name, output):
    print("HERE: ",output)
    metrics = {
        "model": model_name,
        "accuracy": output.get("accuracy", None),
        "f1_score": output.get("f1_score", None),
        "precision": output.get("precision", None),
        "recall": output.get("recall", None),
    }

    return metrics

# User interface
class PipelineDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Pipeline Dashboard")
        self.root.geometry("1400x900")

        self.model_results = {}
        self.metrics_list = []

        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # This part describes the parsers
        parser_frame = ttk.LabelFrame(left_frame, text="1. Parsing")
        parser_frame.pack(fill=tk.X, pady=5)

        self.parser_vars = {}

        available_parsers = discover_modules(PARSER_DIR)

        for parser in available_parsers:
            var = tk.BooleanVar(
                value=parser in ["graphs_to_csv"]
            )

            self.parser_vars[parser] = var

            ttk.Checkbutton(
                parser_frame,
                text=parser,
                variable=var
            ).pack(anchor="w")

        # And this the splitters
        splitter_frame = ttk.LabelFrame(left_frame, text="2. Splitter")
        splitter_frame.pack(fill=tk.X, pady=5)

        self.splitter_var = tk.StringVar()

        available_splitters = discover_modules(SPLITTER_DIR)

        if available_splitters:
            self.splitter_var.set(available_splitters[0])

        for splitter in available_splitters:
            ttk.Radiobutton(
                splitter_frame,
                text=splitter,
                variable=self.splitter_var,
                value=splitter
            ).pack(anchor="w")

        # And finally the available models
        model_frame = ttk.LabelFrame(left_frame, text="3. Models")
        model_frame.pack(fill=tk.X, pady=5)

        self.model_vars = {}

        available_models = discover_modules(MODELS_DIR)

        for model in available_models:
            var = tk.BooleanVar(value=True)

            self.model_vars[model] = var

            ttk.Checkbutton(
                model_frame,
                text=model,
                variable=var
            ).pack(anchor="w")

        # Buttons, progress, etc.
        run_button = ttk.Button(
            left_frame,
            text="Run Pipeline",
            command=self.run_pipeline
        )
        run_button.pack(fill=tk.X, pady=10)

        self.progress = ttk.Progressbar(
            left_frame,
            orient="horizontal",
            mode="determinate"
        )
        self.progress.pack(fill=tk.X, pady=5)

        # Table
        table_frame = ttk.LabelFrame(right_frame, text="Results Table")
        table_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        self.table = ttk.Treeview(table_frame)
        self.table.pack(fill=tk.BOTH, expand=True)

        # Optional scrollbar
        table_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview
        )

        self.table.configure(yscrollcommand=table_scroll.set)

        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # And log
        output_frame = ttk.LabelFrame(right_frame, text="Pipeline Output")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = ScrolledText(output_frame, height=15)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def log(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.root.update()

    # This function runs the entire pipeline after specification in the UI
    def run_pipeline(self):

        self.output_text.delete("1.0", tk.END)

        self.model_results = {}
        self.metrics_list = []

        selected_parsers = [
            name for name, var in self.parser_vars.items()
            if var.get()
        ]

        selected_splitter = self.splitter_var.get()

        selected_models = [
            name for name, var in self.model_vars.items()
            if var.get()
        ]

        total_steps = (
            len(selected_parsers)
            + 1
            + len(selected_models)
        )

        current_step = 0

        self.log("--- RUNNING PARSER STEP ---")

        for parser_name in selected_parsers:

            self.log(f"Running parser: {parser_name}")

            path = PARSER_DIR / f"{parser_name}.py"

            code, parser_output = execute_module(path)

            save_log(parser_name, str(parser_output))

            self.log("Finished parsing step...")

            current_step += 1
            self.progress["value"] = (current_step / total_steps) * 100
            self.root.update()

            if code != 0:
                messagebox.showerror("Parser Error", parser_output)

        self.log("--- RUNNING SPLITTER STEP ---")

        splitter_path = SPLITTER_DIR / f"{selected_splitter}.py"

        code, split_output = execute_module(splitter_path)

        save_log(selected_splitter, str(split_output))

        self.log("Finished splitting step...")

        current_step += 1
        self.progress["value"] = (current_step / total_steps) * 100
        self.root.update()

        if code != 0:
            messagebox.showerror("Splitter Error", split_output)

        self.log("--- RUNNING MODELS STEP ---")

        for model_name in selected_models:

            self.log(f"Running model: {model_name}")

            model_path = MODELS_DIR / f"{model_name}.py"

            start = time.time()

            code, model_output = execute_module(model_path, split_output)
            print("Model Output:", model_output)

            duration = time.time() - start

            save_log(model_name, str(model_output))

            self.model_results[model_name] = {
                "output": model_output,
                "runtime_seconds": duration,
            }

            metrics = extract_metrics(model_name, model_output)
            metrics["runtime_seconds"] = duration

            self.metrics_list.append(metrics)

            self.log("Finished running model(s)...")

            current_step += 1
            self.progress["value"] = (current_step / total_steps) * 100
            self.root.update()

            if code != 0:
                messagebox.showerror("Model Error", model_output)

        self.update_table()

        messagebox.showinfo("Success", "Pipeline completed successfully")

    def update_table(self):

        for item in self.table.get_children():
            self.table.delete(item)

        if not self.metrics_list:
            return

        columns = list(self.metrics_list[0].keys())

        self.table["columns"] = columns
        self.table["show"] = "headings"

        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=120)

        for row in self.metrics_list:
            values = [row.get(c, "") for c in columns]
            self.table.insert("", tk.END, values=values)

        df = pd.DataFrame(self.metrics_list)

        csv_path = REPORT_DIR / "model_comparison.csv"
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineDashboard(root)
    root.mainloop()
