import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.ingest.troubleshooting_completion import build_troubleshooting_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted troubleshooting coverage for weak Financials modules.")
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--no-reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_troubleshooting_completion(tenant_id=args.tenant, reset_index=not args.no_reset)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
