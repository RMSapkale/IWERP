import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.ingest.specialization_tracks import build_specialization_tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    summary = build_specialization_tracks(tenant_id=args.tenant, reset_indexes=not args.no_reset)
    print(json.dumps(summary["asset_inventory"], indent=2))
    print(json.dumps(summary["index_stats"], indent=2))


if __name__ == "__main__":
    main()
