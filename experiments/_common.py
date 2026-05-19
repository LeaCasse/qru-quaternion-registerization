from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
DAT = ROOT / "outputs" / "data"
for p in [FIG, TAB, DAT]:
    p.mkdir(parents=True, exist_ok=True)
