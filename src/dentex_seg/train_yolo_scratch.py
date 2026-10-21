from pathlib import Path
import runpy

if __name__ == "__main__":
    root_train = Path(__file__).resolve().parents[2] / "train.py"
    runpy.run_path(str(root_train), run_name="__main__")
