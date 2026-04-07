import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import CACHE_PATH, get_default_registry

TABLE_INDEX_PATH = BASE_DIR / "backend" / "core" / "grounding" / "fusion_tables_index.json"


def main():
    registry = get_default_registry()
    registry.rebuild()
    with open(TABLE_INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(sorted(registry.objects.keys()), handle, indent=2)
    print(f"Trusted registry rebuilt at {CACHE_PATH}")
    print(f"Table index saved to {TABLE_INDEX_PATH}")


if __name__ == "__main__":
    main()
