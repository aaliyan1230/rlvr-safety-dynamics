#!/usr/bin/env python3
from pathlib import Path
import sys
for candidate in sorted(Path("/kaggle/input").glob("**/src")):
    if (candidate / "rlvr_safety").is_dir(): sys.path.insert(0, str(candidate)); break
from rlvr_safety.kaggle_trajectory import main
if __name__ == "__main__": main(Path(__file__).resolve(), "tulu_grpo_trajectory_refinement_wave_06.json")
