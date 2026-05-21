from pathlib import Path

def find_project_root(marker: str = ".root") -> Path:
    """
    Walk up the directory tree from this file until a directory
    containing `marker` is found. Raises RuntimeError if not found.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Could not find project root. "
        f"Make sure a '{marker}' file exists in your project root."
    )