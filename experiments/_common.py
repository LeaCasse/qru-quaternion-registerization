from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
DATA = ROOT / "outputs" / "data"
META = ROOT / "outputs" / "metadata"
for p in (FIG, TAB, DATA, META):
    p.mkdir(parents=True, exist_ok=True)
