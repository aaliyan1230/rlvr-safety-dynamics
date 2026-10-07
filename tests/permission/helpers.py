from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def scenario() -> dict:
    return deepcopy(json.loads((FIXTURES / "scenario.json").read_text()))


def control() -> dict:
    return deepcopy(json.loads((FIXTURES / "control.json").read_text()))
