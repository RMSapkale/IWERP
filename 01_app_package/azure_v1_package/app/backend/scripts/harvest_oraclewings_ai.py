import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.ingest.oraclewings_ai_harvest import (
    DEFAULT_SOURCE_BRANCH,
    DEFAULT_SOURCE_REPO,
    harvest_oraclewings_ai,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--branch", default=DEFAULT_SOURCE_BRANCH)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--source-dir", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    source_dir = Path(args.source_dir) if args.source_dir else None
    summary = harvest_oraclewings_ai(
        source_repo=args.repo,
        branch=args.branch,
        output_dir=output_dir,
        source_dir=source_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
