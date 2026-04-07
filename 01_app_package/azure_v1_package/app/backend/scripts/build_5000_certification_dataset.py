import json
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")

import sys

sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.evaluation.benchmark_ops import build_expanded_5000_dataset


def main() -> None:
    path = build_expanded_5000_dataset(force=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"dataset_path": str(path), "total_cases": payload.get("total_cases")}, indent=2))


if __name__ == "__main__":
    main()
