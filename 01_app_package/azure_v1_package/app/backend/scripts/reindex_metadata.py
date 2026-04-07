import json
import sys
from pathlib import Path

BASE_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap/iwerp-prod")
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "backend"))

from backend.core.grounding.trusted_registry import CACHE_PATH, get_default_registry

TABLE_INDEX_PATH = BASE_DIR / "backend" / "core" / "grounding" / "fusion_tables_index.json"
AUDIT_REPORT_PATH = BASE_DIR / "backend" / "core" / "grounding" / "trusted_object_registry_audit.json"


def main():
    registry = get_default_registry()
    baseline_unknown = sum(
        1
        for entry in registry.objects.values()
        if (entry.get("owning_module_family") or entry.get("owning_module") or "UNKNOWN") == "UNKNOWN"
    )
    registry.rebuild()

    table_names = sorted(registry.objects.keys())
    with open(TABLE_INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump(table_names, handle, indent=2)
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(registry.module_audit_report(baseline_unknown=baseline_unknown), handle, indent=2, sort_keys=True)

    print(f"Trusted registry rebuilt at {CACHE_PATH}")
    print(f"Table index refreshed at {TABLE_INDEX_PATH}")
    print(f"Audit report saved to {AUDIT_REPORT_PATH}")
    print(f"Registered objects: {len(table_names)}")


if __name__ == "__main__":
    main()
