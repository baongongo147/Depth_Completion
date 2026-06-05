import subprocess
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run test.py for all PyTorch models in checkpoints/models")
    parser.add_argument("--models_dir", type=Path, default=Path("checkpoints/models"))
    parser.add_argument("--rgbd_dir", type=Path, default=Path("Test_Datasets/Private_Test"))
    parser.add_argument("--python", type=str, default="python")
    args = parser.parse_args()

    models = sorted([p for p in args.models_dir.iterdir() if p.suffix == ".pth"])
    if not models:
        print("No .pth models found in", args.models_dir)
        return

    summary = []
    for m in models:
        print(f"\n--- Running model: {m.name} ---")
        cmd = [args.python, "test.py", "--model_dir", str(m), "--rgbd_dir", str(args.rgbd_dir)]
        print(" ", " ".join(cmd))
        res = subprocess.run(cmd)
        summary.append((m.name, res.returncode))

    print("\n--- Summary ---")
    for name, rc in summary:
        print(f"{name}: returncode={rc}")


if __name__ == "__main__":
    main()
