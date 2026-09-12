from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_gold_baselines_v2 as v2
import run_gold_reference_baselines as reference

reference.base.load_gold_qa = v2.load_gold_qa

if __name__ == '__main__':
    raise SystemExit(reference.main())
