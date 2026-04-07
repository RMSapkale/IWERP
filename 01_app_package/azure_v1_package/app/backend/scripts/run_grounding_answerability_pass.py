import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.ingest.grounding_answerability_pass import run_grounding_answerability_pass


if __name__ == "__main__":
    summary = run_grounding_answerability_pass()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
