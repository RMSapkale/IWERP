import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import (
    TAXONOMY_CONFLICT_PATH,
    TAXONOMY_OVERRIDE_PATH,
    TAXONOMY_SEED_PATH,
)


def main():
    seed = json.loads(TAXONOMY_SEED_PATH.read_text(encoding="utf-8"))
    conflicts = json.loads(TAXONOMY_CONFLICT_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(TAXONOMY_OVERRIDE_PATH.read_text(encoding="utf-8"))

    with open(TAXONOMY_SEED_PATH, "w", encoding="utf-8") as handle:
        json.dump(seed, handle, indent=2, sort_keys=True)
    with open(TAXONOMY_CONFLICT_PATH, "w", encoding="utf-8") as handle:
        json.dump(conflicts, handle, indent=2, sort_keys=True)
    with open(TAXONOMY_OVERRIDE_PATH, "w", encoding="utf-8") as handle:
        json.dump(overrides, handle, indent=2, sort_keys=True)

    print(f"Taxonomy seed refreshed at {TAXONOMY_SEED_PATH}")
    print(f"Taxonomy conflicts refreshed at {TAXONOMY_CONFLICT_PATH}")
    print(f"Manual overrides refreshed at {TAXONOMY_OVERRIDE_PATH}")
    print(f"Seed authority note: {seed.get('seed_note')}")


if __name__ == "__main__":
    main()
