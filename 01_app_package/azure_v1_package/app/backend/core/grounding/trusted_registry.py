import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

import structlog

from core.schemas.curation import RegistryObjectType, TrustedObjectEntry
from core.schemas.router import FusionModule, ModuleFamily, module_families_for_value

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent
METADATA_DIR = BASE_DIR / "metadata" / "csvs"
MAPPING_PATH = BASE_DIR / "fusion_ebs_mapping.json"
CACHE_PATH = BASE_DIR / "trusted_object_registry.json"
CACHE_VERSION = 4
TAXONOMY_SEED_PATH = BASE_DIR / "oracle_fusion_taxonomy_seed.json"
TAXONOMY_CONFLICT_PATH = BASE_DIR / "oracle_fusion_taxonomy_conflicts.json"
TAXONOMY_OVERRIDE_PATH = BASE_DIR / "oracle_fusion_taxonomy_manual_overrides.json"
OBJECT_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$]{2,}$")
COLUMN_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$]{1,}$")
MODULE_CONFIDENCE_THRESHOLD = 0.75
SOURCE_FILE_FAMILY_HINTS: Dict[str, str] = {
    "FUSION_EXTRACTED_FKS_SCM.csv": ModuleFamily.SCM.value,
    "FUSION_EXTRACTED_FKS_PROC.csv": ModuleFamily.PROCUREMENT.value,
    "FUSION_EXTRACTED_FKS_PROJ.csv": ModuleFamily.PROJECTS.value,
}
SOURCE_HINT_MODULE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(accounts payable|\bAP\b|payables|supplier)"), FusionModule.PAYABLES.value),
    (re.compile(r"(?i)(accounts receivable|\bAR\b|receivables|receipt|customer)"), FusionModule.RECEIVABLES.value),
    (re.compile(r"(?i)(general ledger|\bGL\b|journal|ledger)"), FusionModule.GENERAL_LEDGER.value),
    (re.compile(r"(?i)(cash management|bank|reconciliation|\bCE\b)"), FusionModule.CASH_MANAGEMENT.value),
    (re.compile(r"(?i)(asset|depreciation|\bFA\b)"), FusionModule.ASSETS.value),
    (re.compile(r"(?i)(procurement|purchasing|supplier portal|requisition|\bPO\b|\bPOZ\b|\bPOR\b)"), FusionModule.PROCUREMENT.value),
    (re.compile(r"(?i)(inventory|shipping|order management|manufacturing|planning|supply chain|\bINV\b|\bDOO\b|\bMSC\b|\bWSH\b)"), FusionModule.SCM.value),
    (re.compile(r"(?i)(hcm|payroll|benefit|absence|employee|fast formula|\bPER\b|\bPAY\b)"), FusionModule.HCM.value),
    (re.compile(r"(?i)(projects|ppm|grant|\bPJT\b|\bPJC\b|\bPJF\b)"), FusionModule.PROJECTS.value),
    (re.compile(r"(?i)(tax|\bZX\b)"), FusionModule.TAX.value),
]

LEGACY_EXACT_MODULE_PREFIXES: Dict[str, str] = {
    "AP_": FusionModule.PAYABLES.value,
    "POZ_": FusionModule.PAYABLES.value,
    "AR_": FusionModule.RECEIVABLES.value,
    "RA_": FusionModule.RECEIVABLES.value,
    "HZ_": FusionModule.RECEIVABLES.value,
    "GL_": FusionModule.GENERAL_LEDGER.value,
    "CE_": FusionModule.CASH_MANAGEMENT.value,
    "FA_": FusionModule.ASSETS.value,
    "EXM_": FusionModule.EXPENSES.value,
    "PO_": FusionModule.PROCUREMENT.value,
    "POR_": FusionModule.PROCUREMENT.value,
    "PON_": FusionModule.PROCUREMENT.value,
    "INV_": FusionModule.SCM.value,
    "WSH_": FusionModule.SCM.value,
    "DOO_": FusionModule.SCM.value,
    "EGP_": FusionModule.SCM.value,
    "PER_": FusionModule.HCM.value,
    "PAY_": FusionModule.HCM.value,
    "HRC_": FusionModule.HCM.value,
    "PJT": FusionModule.PROJECTS.value,
    "PJC": FusionModule.PROJECTS.value,
    "PJF": FusionModule.PROJECTS.value,
    "PA_": FusionModule.PROJECTS.value,
    "ZX_": FusionModule.TAX.value,
}

SQL_INDEX_ROOT = BASE_DIR.parent / "retrieval" / "vectors" / "faiss"
PROJECT_DIR = BASE_DIR.parents[2]
SQL_MANIFEST_PATH = PROJECT_DIR / "specialization_tracks" / "manifests" / "sql_examples_manifest.jsonl"


class TrustedObjectRegistry:
    _default_instance: Optional["TrustedObjectRegistry"] = None

    def __init__(self):
        self.objects: Dict[str, Dict[str, Any]] = {}
        self.columns_by_table: Dict[str, Set[str]] = {}
        self.alias_to_object: Dict[str, str] = {}
        self.relation_details_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.join_graph: Dict[str, Set[str]] = defaultdict(set)
        self.view_base_tables: Dict[str, List[str]] = {}
        self.ebs_mappings: Dict[str, str] = {}
        self.taxonomy_seed: Dict[str, Any] = {}
        self.taxonomy_conflicts: Dict[str, Any] = {}
        self.manual_overrides: Dict[str, Any] = {}
        self._seed_prefix_to_family: Dict[str, str] = {}
        self._seed_leaf_hints: Dict[str, str] = {}
        self._weak_prefix_to_family: Dict[str, str] = {}
        self._weak_leaf_hints: Dict[str, str] = {}
        self._conflicting_prefixes: Set[str] = set()
        self._sql_family_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        self._sql_module_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        self._docs_family_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        self._docs_module_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        self._usage_family_votes: Dict[str, Counter[str]] = defaultdict(Counter)
        self._usage_module_votes: Dict[str, Counter[str]] = defaultdict(Counter)

    @classmethod
    def get_default(cls) -> "TrustedObjectRegistry":
        if cls._default_instance is None:
            cls._default_instance = cls()
            cls._default_instance.load()
        return cls._default_instance

    def load(self) -> None:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if int(payload.get("cache_version") or 0) != CACHE_VERSION:
                    raise ValueError("registry_cache_version_mismatch")
                self.taxonomy_seed = payload.get("taxonomy_seed", self._load_taxonomy_seed())
                self.taxonomy_conflicts = payload.get("taxonomy_conflicts", self._load_taxonomy_conflicts())
                self.manual_overrides = payload.get("manual_overrides", self._load_manual_overrides())
                self._build_taxonomy_indexes()
                self.objects = {
                    name: self._normalize_loaded_entry(entry)
                    for name, entry in payload.get("objects", {}).items()
                }
                self.columns_by_table = {
                    table: set(columns) for table, columns in payload.get("columns_by_table", {}).items()
                }
                self.alias_to_object = payload.get("alias_to_object", {})
                self.ebs_mappings = payload.get("ebs_mappings", {})
                self._reindex_runtime_metadata()
                logger.info("trusted_registry_loaded", objects=len(self.objects), path=str(CACHE_PATH))
                return
            except Exception as exc:
                logger.warning("trusted_registry_cache_load_failed", error=str(exc))

        self.rebuild()

    def rebuild(self) -> None:
        self.objects = {}
        self.columns_by_table = {}
        self.alias_to_object = {}
        self.ebs_mappings = self._load_ebs_mappings()
        self.taxonomy_seed = self._load_taxonomy_seed()
        self.taxonomy_conflicts = self._load_taxonomy_conflicts()
        self.manual_overrides = self._load_manual_overrides()
        self._build_taxonomy_indexes()

        self._load_tables()
        self._load_views()
        self._load_columns()
        self._load_primary_keys()
        self._load_relations()
        self._load_synonyms()
        self._apply_ebs_aliases()
        self._load_sql_votes()
        self._load_docs_votes()
        self._load_usage_votes()
        self._apply_manual_overrides()
        self._apply_module_inference()
        self._reindex_runtime_metadata()

        payload = {
            "cache_version": CACHE_VERSION,
            "objects": self.objects,
            "columns_by_table": {table: sorted(columns) for table, columns in self.columns_by_table.items()},
            "alias_to_object": self.alias_to_object,
            "ebs_mappings": self.ebs_mappings,
            "taxonomy_seed": self.taxonomy_seed,
            "taxonomy_conflicts": self.taxonomy_conflicts,
            "manual_overrides": self.manual_overrides,
        }
        with open(CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        logger.info("trusted_registry_rebuilt", objects=len(self.objects), path=str(CACHE_PATH))

    def _build_taxonomy_indexes(self) -> None:
        self._seed_prefix_to_family = {}
        self._seed_leaf_hints = {
            str(prefix).strip().upper(): str(module).strip()
            for prefix, module in (self.taxonomy_seed.get("leaf_hints") or {}).items()
            if str(prefix).strip()
        }
        self._weak_prefix_to_family = {
            str(prefix).strip().upper(): str(family).strip()
            for prefix, family in (self.taxonomy_seed.get("weak_families") or {}).items()
            if str(prefix).strip()
        }
        self._weak_leaf_hints = {
            str(prefix).strip().upper(): str(module).strip()
            for prefix, module in (self.taxonomy_seed.get("weak_leaf_hints") or {}).items()
            if str(prefix).strip()
        }
        self._conflicting_prefixes = {
            str(prefix).strip().upper()
            for prefix in (self.taxonomy_conflicts.get("prefix_conflicts") or {}).keys()
            if str(prefix).strip()
        }

        for family_name, prefixes in (self.taxonomy_seed.get("families") or {}).items():
            for prefix in prefixes:
                prefix_key = str(prefix).strip().upper()
                if not prefix_key or prefix_key in self._conflicting_prefixes:
                    continue
                self._seed_prefix_to_family[prefix_key] = str(family_name).strip()

    def _normalize_loaded_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(entry)
        owning_module = str(normalized.get("owning_module") or FusionModule.UNKNOWN.value)
        family = str(normalized.get("owning_module_family") or self._family_for_exact_module(owning_module))
        inferred = str(normalized.get("inferred_module") or owning_module)
        normalized.setdefault("original_owning_module", owning_module)
        normalized.setdefault("owning_module_family", family)
        normalized.setdefault("inferred_module", inferred)
        normalized.setdefault("confidence_score", 1.0 if owning_module != FusionModule.UNKNOWN.value else 0.0)
        normalized.setdefault("inference_source", "loaded")
        normalized.setdefault("low_confidence", normalized["confidence_score"] < MODULE_CONFIDENCE_THRESHOLD)
        normalized.setdefault("manual_lock", False)
        normalized.setdefault("relation_details", [])
        normalized.setdefault("primary_keys", [])
        normalized.setdefault("base_tables", [])
        return TrustedObjectEntry(**normalized).model_dump(mode="json")

    def _ensure_object(self, object_name: str, object_type: RegistryObjectType, source: str, confidence: float = 1.0) -> None:
        canonical = self._canonicalize(object_name)
        if not canonical:
            return

        default_exact = self._infer_legacy_exact_module(canonical)
        default_family = self._family_for_exact_module(default_exact)
        entry = self.objects.setdefault(
            canonical,
            TrustedObjectEntry(
                object_name=canonical,
                object_type=object_type,
                owning_module=default_exact,
                original_owning_module=default_exact,
                owning_module_family=default_family,
                inferred_module=default_exact,
                confidence_score=1.0 if default_exact != FusionModule.UNKNOWN.value else 0.0,
                inference_source="legacy_prefix" if default_exact != FusionModule.UNKNOWN.value else "none",
                low_confidence=default_exact == FusionModule.UNKNOWN.value,
                manual_lock=False,
                aliases=[],
                ebs_aliases=[],
                approved_relations=[],
                relation_details=[],
                source_of_truth=[source],
                confidence=confidence,
                columns=[],
                primary_keys=[],
                base_tables=[],
            ).model_dump(mode="json"),
        )
        if source not in entry["source_of_truth"]:
            entry["source_of_truth"].append(source)
        entry["confidence"] = max(entry.get("confidence", 0.0), confidence)
        if entry["object_type"] != object_type.value and object_type == RegistryObjectType.TABLE:
            entry["object_type"] = object_type.value
        self.alias_to_object[canonical] = canonical

    def _canonicalize(self, raw_name: Optional[str]) -> Optional[str]:
        if not raw_name:
            return None
        candidate = str(raw_name).strip().upper()
        if not OBJECT_PATTERN.match(candidate):
            return None
        return candidate

    def _canonicalize_column(self, raw_name: Optional[str]) -> Optional[str]:
        if not raw_name:
            return None
        candidate = str(raw_name).strip().upper()
        if not COLUMN_PATTERN.match(candidate):
            return None
        return candidate

    def _load_ebs_mappings(self) -> Dict[str, str]:
        if not MAPPING_PATH.exists():
            return {}
        with open(MAPPING_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            self._canonicalize(k): self._canonicalize(v)
            for k, v in payload.get("mappings", {}).items()
            if self._canonicalize(k) and self._canonicalize(v)
        }

    def _load_taxonomy_seed(self) -> Dict[str, Any]:
        if not TAXONOMY_SEED_PATH.exists():
            return {"sheet_url": "", "families": {}, "leaf_hints": {}}
        with open(TAXONOMY_SEED_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_taxonomy_conflicts(self) -> Dict[str, Any]:
        if not TAXONOMY_CONFLICT_PATH.exists():
            return {"prefix_conflicts": {}}
        with open(TAXONOMY_CONFLICT_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_manual_overrides(self) -> Dict[str, Any]:
        if not TAXONOMY_OVERRIDE_PATH.exists():
            return {"objects": {}}
        with open(TAXONOMY_OVERRIDE_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_standard_rows(self, path: Path) -> Iterable[Dict[str, str]]:
        with open(path, newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield {str(key).strip().upper(): str(value).strip() for key, value in row.items() if key}

    def _read_pipe_rows(self, path: Path) -> Iterable[Dict[str, str]]:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().lstrip("\ufeff").strip().split("|")
            upper_header = [column.strip().upper() for column in header]
            for line in handle:
                parts = [part.strip() for part in line.rstrip("\n").split("|")]
                if not parts or len(parts) != len(upper_header):
                    continue
                yield dict(zip(upper_header, parts))

    def _append_relation_detail(
        self,
        table_name: str,
        related_table: str,
        *,
        local_column: Optional[str],
        related_column: Optional[str],
        fk_name: str,
        position: str,
        source_name: str,
        direction: str,
    ) -> None:
        entry = self.objects.get(table_name)
        if not entry:
            return
        detail = {
            "related_table": related_table,
            "local_column": local_column or "",
            "related_column": related_column or "",
            "fk_name": fk_name,
            "position": position,
            "source_name": source_name,
            "direction": direction,
        }
        if detail not in entry["relation_details"]:
            entry["relation_details"].append(detail)

    def _record_relation_pair(
        self,
        src_table: str,
        tgt_table: str,
        *,
        src_column: Optional[str],
        tgt_column: Optional[str],
        fk_name: str,
        position: str,
        source_name: str,
    ) -> None:
        pair = tuple(sorted((src_table, tgt_table)))
        detail = {
            "source_table": src_table,
            "source_column": src_column or "",
            "target_table": tgt_table,
            "target_column": tgt_column or "",
            "fk_name": fk_name,
            "position": position,
            "source_name": source_name,
        }
        existing = self.relation_details_by_pair[pair]
        if detail not in existing:
            existing.append(detail)
        self.join_graph[src_table].add(tgt_table)
        self.join_graph[tgt_table].add(src_table)

    def _reindex_runtime_metadata(self) -> None:
        self.relation_details_by_pair = defaultdict(list)
        self.join_graph = defaultdict(set)
        self.view_base_tables = {}
        for object_name, entry in self.objects.items():
            for related_name in entry.get("approved_relations", []):
                self.join_graph[object_name].add(related_name)
            base_tables = [str(table).upper() for table in entry.get("base_tables", []) if self._canonicalize(table)]
            if base_tables:
                self.view_base_tables[object_name] = base_tables
                for base_table in base_tables:
                    self.join_graph[object_name].add(base_table)
                    self.join_graph[base_table].add(object_name)
            for detail in entry.get("relation_details", []):
                related_table = self._canonicalize(detail.get("related_table"))
                local_column = self._canonicalize_column(detail.get("local_column"))
                related_column = self._canonicalize_column(detail.get("related_column"))
                if not related_table:
                    continue
                pair = tuple(sorted((object_name, related_table)))
                pair_detail = {
                    "source_table": object_name if detail.get("direction") == "outbound" else related_table,
                    "source_column": local_column if detail.get("direction") == "outbound" else related_column or "",
                    "target_table": related_table if detail.get("direction") == "outbound" else object_name,
                    "target_column": related_column if detail.get("direction") == "outbound" else local_column or "",
                    "fk_name": str(detail.get("fk_name") or ""),
                    "position": str(detail.get("position") or ""),
                    "source_name": str(detail.get("source_name") or ""),
                }
                if pair_detail not in self.relation_details_by_pair[pair]:
                    self.relation_details_by_pair[pair].append(pair_detail)
                self.join_graph[object_name].add(related_table)
                self.join_graph[related_table].add(object_name)

    def _extract_view_base_tables(self, definition_sql: str) -> List[str]:
        sql_text = (definition_sql or "").strip()
        if not sql_text:
            return []
        try:
            tree = sqlglot.parse_one(sql_text, read="oracle")
            candidates = {
                self._canonicalize(table.name)
                for table in tree.find_all(exp.Table)
                if self._canonicalize(table.name)
            }
            return sorted(candidates)
        except Exception:
            tokens = {
                self._canonicalize(token)
                for token in re.findall(r'"([A-Za-z][A-Za-z0-9_$]{2,})"', sql_text)
                if self._canonicalize(token)
            }
            return sorted(tokens)

    def _extract_view_select_columns(self, definition_sql: str) -> List[str]:
        sql_text = (definition_sql or "").strip()
        if not sql_text:
            return []

        columns: Set[str] = set()
        try:
            tree = sqlglot.parse_one(sql_text, read="oracle")
            if isinstance(tree, exp.Select):
                select_exprs = list(tree.expressions or [])
            else:
                select_exprs = list(getattr(tree, "expressions", []) or [])

            for select_expr in select_exprs:
                alias_or_name = ""
                try:
                    alias_or_name = str(select_expr.alias_or_name or "").strip()
                except Exception:
                    alias_or_name = ""

                if alias_or_name:
                    canonical = self._canonicalize_column(alias_or_name)
                    if canonical:
                        columns.add(canonical)
                    continue

                if isinstance(select_expr, exp.Column):
                    canonical = self._canonicalize_column(select_expr.name)
                    if canonical:
                        columns.add(canonical)
        except Exception:
            pass

        if columns:
            return sorted(columns)

        # Fallback for malformed definitions: quoted projection list before FROM.
        match = re.search(r"(?is)\bSELECT\b(.*?)\bFROM\b", sql_text)
        if not match:
            return []
        projection_sql = match.group(1)
        for token in re.findall(r'"([A-Za-z][A-Za-z0-9_$]{1,})"', projection_sql):
            canonical = self._canonicalize_column(token)
            if canonical:
                columns.add(canonical)
        return sorted(columns)

    def _load_tables(self) -> None:
        for path in [METADATA_DIR / "ALL_TABLES.csv", METADATA_DIR / "metadata_ALL_TABLES.csv"]:
            if not path.exists():
                continue
            for row in self._read_standard_rows(path):
                table_name = self._canonicalize(row.get("TABLE_NAME"))
                if table_name:
                    self._ensure_object(table_name, RegistryObjectType.TABLE, path.name, confidence=1.0)

    def _load_views(self) -> None:
        for path in [METADATA_DIR / "ALL_VIEWS_RPT.csv", METADATA_DIR / "metadata_ALL_VIEWS_RPT.csv"]:
            if not path.exists():
                continue
            for row in self._read_standard_rows(path):
                view_name = self._canonicalize(row.get("VIEW_NAME"))
                if not view_name:
                    continue
                self._ensure_object(view_name, RegistryObjectType.VIEW, path.name, confidence=0.95)
                definition_sql = str(row.get("TEXT_VC") or row.get("TEXT") or "").strip()
                if not definition_sql:
                    continue
                base_tables = self._extract_view_base_tables(definition_sql)
                if not base_tables:
                    base_tables = []

                entry = self.objects[view_name]
                if base_tables:
                    entry["base_tables"] = sorted(set(entry.get("base_tables", [])) | set(base_tables))
                    for base_table in base_tables:
                        self._ensure_object(base_table, RegistryObjectType.TABLE, path.name, confidence=0.9)
                        if base_table not in entry["approved_relations"]:
                            entry["approved_relations"].append(base_table)
                        related_entry = self.objects.get(base_table)
                        if related_entry and view_name not in related_entry["approved_relations"]:
                            related_entry["approved_relations"].append(view_name)

                view_columns = self._extract_view_select_columns(definition_sql)
                if view_columns:
                    existing = set(entry.get("columns", []))
                    existing.update(view_columns)
                    entry["columns"] = sorted(existing)
                    self.columns_by_table.setdefault(view_name, set()).update(view_columns)

    def _load_primary_keys(self) -> None:
        candidates: Dict[str, List[Tuple[int, List[str], str]]] = defaultdict(list)
        for path in METADATA_DIR.glob("FUSION_EXTRACTED_INDEXES_*.csv"):
            rows = self._read_pipe_rows(path)
            for row in rows:
                table_name = self._canonicalize(row.get("TABLE_NAME"))
                index_name = str(row.get("INDEX_NAME") or "").strip().upper()
                uniqueness = str(row.get("UNIQUENESS") or "").strip().upper()
                if not table_name or uniqueness != "UNIQUE":
                    continue
                raw_columns = str(row.get("COLUMNS") or "").strip()
                if not raw_columns:
                    raw_columns = str(row.get("TABLESPACE") or "").strip()
                columns = [
                    canonical
                    for canonical in (
                        self._canonicalize_column(part.strip().strip('"'))
                        for part in raw_columns.split(",")
                    )
                    if canonical and not canonical.startswith("ORA_SEED_")
                ]
                if not columns:
                    continue
                score = 3 if "_PK" in index_name or index_name.endswith("PK") else 2
                score += max(0, 4 - min(len(columns), 4))
                candidates[table_name].append((score, columns, index_name))

        for table_name, entries in candidates.items():
            entries.sort(key=lambda item: (-item[0], len(item[1]), item[2]))
            best_columns = entries[0][1]
            entry = self.objects.get(table_name)
            if entry:
                entry["primary_keys"] = best_columns

    def _load_columns(self) -> None:
        for path in [METADATA_DIR / "ALL_TAB_COLUMNS.csv", METADATA_DIR / "metadata_ALL_TAB_COLUMNS.csv", METADATA_DIR / "all_tab_columns_rpt.csv"]:
            if not path.exists():
                continue
            rows = self._read_pipe_rows(path) if path.name.endswith("_rpt.csv") else self._read_standard_rows(path)
            for row in rows:
                table_name = self._canonicalize(row.get("TABLE_NAME"))
                raw_column_name = str(row.get("COLUMN_NAME") or "").strip()
                column_name = self._canonicalize_column(raw_column_name)
                if not table_name:
                    continue
                self._ensure_object(table_name, RegistryObjectType.TABLE, path.name, confidence=1.0)
                if column_name:
                    self.columns_by_table.setdefault(table_name, set()).add(column_name)
                    continue

                # Some metadata dumps store multiple columns in one cell for views.
                if " " not in raw_column_name and "," not in raw_column_name:
                    continue
                token_candidates = [
                    self._canonicalize_column(token.strip().strip('"'))
                    for token in re.split(r"[\s,]+", raw_column_name)
                    if token.strip()
                ]
                expanded_columns = [
                    candidate
                    for candidate in token_candidates
                    if candidate and not candidate.startswith("ORA_SEED_")
                ]
                if len(expanded_columns) < 2:
                    continue
                self.columns_by_table.setdefault(table_name, set()).update(expanded_columns)

        for table_name, columns in self.columns_by_table.items():
            entry = self.objects.get(table_name)
            if entry:
                entry["columns"] = sorted(columns)

    def _load_relations(self) -> None:
        for path in METADATA_DIR.glob("FUSION_EXTRACTED_FKS_*.csv"):
            rows = self._read_pipe_rows(path)
            for row in rows:
                src_table = self._canonicalize(row.get("SRC_TABLE"))
                tgt_table = self._canonicalize(row.get("TGT_TABLE"))
                src_column = self._canonicalize_column(row.get("SRC_COLUMN"))
                tgt_column = self._canonicalize_column(row.get("TGT_COLUMN"))
                fk_name = str(row.get("FK_NAME") or "").strip().upper()
                position = str(row.get("POSITION") or "").strip()
                if not src_table or not tgt_table:
                    continue
                self._ensure_object(src_table, RegistryObjectType.TABLE, path.name, confidence=0.9)
                self._ensure_object(tgt_table, RegistryObjectType.TABLE, path.name, confidence=0.9)
                if tgt_table not in self.objects[src_table]["approved_relations"]:
                    self.objects[src_table]["approved_relations"].append(tgt_table)
                if src_table not in self.objects[tgt_table]["approved_relations"]:
                    self.objects[tgt_table]["approved_relations"].append(src_table)
                if src_column == "UNKNOWN":
                    src_column = None
                if tgt_column == "UNKNOWN":
                    tgt_column = None
                self._append_relation_detail(
                    src_table,
                    tgt_table,
                    local_column=src_column,
                    related_column=tgt_column,
                    fk_name=fk_name,
                    position=position,
                    source_name=path.name,
                    direction="outbound",
                )
                self._append_relation_detail(
                    tgt_table,
                    src_table,
                    local_column=tgt_column,
                    related_column=src_column,
                    fk_name=fk_name,
                    position=position,
                    source_name=path.name,
                    direction="inbound",
                )
                self._record_relation_pair(
                    src_table,
                    tgt_table,
                    src_column=src_column,
                    tgt_column=tgt_column,
                    fk_name=fk_name,
                    position=position,
                    source_name=path.name,
                )

    def _load_synonyms(self) -> None:
        path = METADATA_DIR / "ALL_SYNONYMS.csv"
        if not path.exists():
            return
        for row in self._read_pipe_rows(path):
            synonym = self._canonicalize(row.get("SYNONYM_NAME"))
            table_name = self._canonicalize(row.get("TABLE_NAME"))
            if not synonym or not table_name or table_name not in self.objects:
                continue
            aliases = self.objects[table_name]["aliases"]
            if synonym not in aliases:
                aliases.append(synonym)
            self.alias_to_object[synonym] = table_name

    def _apply_ebs_aliases(self) -> None:
        for ebs_name, fusion_name in self.ebs_mappings.items():
            if fusion_name not in self.objects:
                continue
            ebs_aliases = self.objects[fusion_name]["ebs_aliases"]
            if ebs_name not in ebs_aliases:
                ebs_aliases.append(ebs_name)
            self.alias_to_object[ebs_name] = fusion_name

    def _infer_legacy_exact_module(self, object_name: str) -> str:
        for prefix, module in LEGACY_EXACT_MODULE_PREFIXES.items():
            if object_name.startswith(prefix):
                return module
        return FusionModule.COMMON.value if object_name.startswith("FND_") else FusionModule.UNKNOWN.value

    def _family_for_exact_module(self, module: Optional[str]) -> str:
        families = module_families_for_value(module)
        if families:
            return next(iter(families))
        return ModuleFamily.UNKNOWN.value

    def _seed_family_for_object(self, object_name: str) -> Optional[str]:
        prefix = object_name.split("_", 1)[0]
        if prefix in self._conflicting_prefixes:
            return None
        return self._seed_prefix_to_family.get(prefix)

    def _seed_leaf_for_object(self, object_name: str) -> Optional[str]:
        prefix = object_name.split("_", 1)[0]
        return self._seed_leaf_hints.get(prefix)

    def _load_sql_votes(self) -> None:
        self._sql_family_votes = defaultdict(Counter)
        self._sql_module_votes = defaultdict(Counter)

        if not SQL_INDEX_ROOT.exists():
            return

        for db_path in SQL_INDEX_ROOT.glob("*/sql_corpus/metadata.sqlite"):
            try:
                with sqlite3.connect(db_path) as conn:
                    rows = conn.execute("SELECT metadata FROM chunks").fetchall()
            except Exception as exc:
                logger.warning("sql_vote_load_failed", path=str(db_path), error=str(exc))
                continue

            for (metadata_json,) in rows:
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    continue
                module = str(metadata.get("module") or "").strip()
                family = self._family_for_exact_module(module)
                trusted_objects = metadata.get("trusted_schema_objects") or []
                for object_name in trusted_objects:
                    canonical = self._canonicalize(object_name)
                    if not canonical:
                        continue
                    if module:
                        self._sql_module_votes[canonical][module] += 1
                    if family != ModuleFamily.UNKNOWN.value:
                        self._sql_family_votes[canonical][family] += 1

        if SQL_MANIFEST_PATH.exists():
            try:
                with open(SQL_MANIFEST_PATH, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        metadata = row.get("metadata") or {}
                        module = str(
                            metadata.get("module")
                            or row.get("module")
                            or ""
                        ).strip()
                        family = self._family_for_exact_module(module)
                        source_hint = " ".join(
                            [
                                str(metadata.get("source_uri") or ""),
                                str(metadata.get("source_file") or ""),
                                str(row.get("source_path") or ""),
                                str(metadata.get("title") or row.get("title") or ""),
                            ]
                        )
                        hinted_modules = {
                            module_hint
                            for pattern, module_hint in SOURCE_HINT_MODULE_PATTERNS
                            if pattern.search(source_hint)
                        }
                        trusted_objects = {
                            self._canonicalize(object_name)
                            for object_name in (metadata.get("trusted_schema_objects") or metadata.get("tables_used") or [])
                            if self._canonicalize(object_name)
                        }
                        for canonical in trusted_objects:
                            if module:
                                self._sql_module_votes[canonical][module] += 1
                            if family != ModuleFamily.UNKNOWN.value:
                                self._sql_family_votes[canonical][family] += 1
                            for hinted_module in hinted_modules:
                                hinted_family = self._family_for_exact_module(hinted_module)
                                self._sql_module_votes[canonical][hinted_module] += 2
                                if hinted_family != ModuleFamily.UNKNOWN.value:
                                    self._sql_family_votes[canonical][hinted_family] += 2
            except Exception as exc:
                logger.warning("sql_manifest_vote_load_failed", path=str(SQL_MANIFEST_PATH), error=str(exc))

    def _load_docs_votes(self) -> None:
        self._docs_family_votes = defaultdict(Counter)
        self._docs_module_votes = defaultdict(Counter)
        doc_db = SQL_INDEX_ROOT / "demo" / "docs_corpus" / "metadata.sqlite"
        if not doc_db.exists():
            return

        object_names = set(self.objects.keys())
        token_pattern = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
        try:
            with sqlite3.connect(doc_db) as conn:
                rows = conn.execute("SELECT content, metadata FROM chunks").fetchall()
        except Exception as exc:
            logger.warning("docs_vote_load_failed", path=str(doc_db), error=str(exc))
            return

        for content, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json)
            except Exception:
                continue

            module = str(metadata.get("module") or "").strip()
            family = str(metadata.get("module_family") or self._family_for_exact_module(module))
            if family == ModuleFamily.UNKNOWN.value:
                continue

            trusted_objects = {
                self._canonicalize(object_name)
                for object_name in (metadata.get("trusted_schema_objects") or [])
                if self._canonicalize(object_name)
            }
            tokens = {
                token
                for token in token_pattern.findall(((content or "") + "\n" + str(metadata.get("title") or "")).upper())
                if token in object_names
            }
            for canonical in trusted_objects.union(tokens):
                self._docs_family_votes[canonical][family] += 1
                if module:
                    self._docs_module_votes[canonical][module] += 1

    def _load_usage_votes(self) -> None:
        self._usage_family_votes = defaultdict(Counter)
        self._usage_module_votes = defaultdict(Counter)

        signal_paths = sorted(PROJECT_DIR.glob("*validation*.jsonl"))
        for path in signal_paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue

                        verifier_passed = bool(
                            payload.get("verifier_passed")
                            or payload.get("verification_status") == "PASSED"
                            or payload.get("verifier_status") == "PASSED"
                        )
                        if not verifier_passed:
                            continue

                        module = str(
                            payload.get("module_detected")
                            or payload.get("detected_module")
                            or payload.get("module")
                            or ""
                        ).strip()
                        family = str(payload.get("module_family_detected") or self._family_for_exact_module(module))
                        if family == ModuleFamily.UNKNOWN.value:
                            continue

                        for object_name in (payload.get("schema_objects_used") or []):
                            canonical = self._canonicalize(object_name)
                            if not canonical:
                                continue
                            self._usage_family_votes[canonical][family] += 1
                            if module:
                                self._usage_module_votes[canonical][module] += 1
            except Exception as exc:
                logger.warning("usage_vote_load_failed", path=str(path), error=str(exc))

    def _apply_manual_overrides(self) -> None:
        object_overrides = (self.manual_overrides.get("objects") or {})
        for object_name, override in object_overrides.items():
            canonical = self._canonicalize(object_name)
            if not canonical or canonical not in self.objects:
                continue
            entry = self.objects[canonical]
            if override.get("owning_module"):
                entry["owning_module"] = str(override["owning_module"])
            if override.get("owning_module_family"):
                entry["owning_module_family"] = str(override["owning_module_family"])
            if override.get("inferred_module"):
                entry["inferred_module"] = str(override["inferred_module"])
            entry["original_owning_module"] = entry.get("original_owning_module") or entry.get("owning_module", FusionModule.UNKNOWN.value)
            entry["confidence_score"] = float(override.get("confidence_score", 1.0))
            entry["inference_source"] = str(override.get("inference_source", "manual"))
            entry["manual_lock"] = bool(override.get("manual_lock", True))
            entry["low_confidence"] = bool(override.get("low_confidence", entry["confidence_score"] < MODULE_CONFIDENCE_THRESHOLD))

    def _strong_sql_family_vote(self, object_name: str) -> Tuple[Optional[str], Optional[str]]:
        family_votes = self._sql_family_votes.get(object_name, Counter())
        if not family_votes:
            return None, None
        total = sum(family_votes.values())
        family, count = family_votes.most_common(1)[0]
        if total < 3 or (count / total) < 0.7:
            return None, None

        module_votes = self._sql_module_votes.get(object_name, Counter())
        inferred_module = module_votes.most_common(1)[0][0] if module_votes else family
        return family, inferred_module

    def _strong_docs_family_vote(self, object_name: str) -> Tuple[Optional[str], Optional[str], float]:
        family_votes = self._docs_family_votes.get(object_name, Counter())
        if not family_votes:
            return None, None, 0.0

        total = sum(family_votes.values())
        family, count = family_votes.most_common(1)[0]
        if total == 1:
            score = 0.76
        elif (count / total) >= 0.7:
            score = 0.82
        else:
            return None, None, 0.0

        module_votes = self._docs_module_votes.get(object_name, Counter())
        inferred_module = module_votes.most_common(1)[0][0] if module_votes else family
        return family, inferred_module, score

    def _strong_usage_family_vote(self, object_name: str) -> Tuple[Optional[str], Optional[str], float]:
        family_votes = self._usage_family_votes.get(object_name, Counter())
        if not family_votes:
            return None, None, 0.0

        total = sum(family_votes.values())
        family, count = family_votes.most_common(1)[0]
        if total == 1:
            score = 0.84
        elif (count / total) >= 0.7:
            score = 0.9
        else:
            return None, None, 0.0

        module_votes = self._usage_module_votes.get(object_name, Counter())
        inferred_module = module_votes.most_common(1)[0][0] if module_votes else family
        return family, inferred_module, score

    def _weak_prefix_family_vote(self, object_name: str) -> Tuple[Optional[str], Optional[str], float]:
        prefix = object_name.split("_", 1)[0]
        if prefix in self._conflicting_prefixes:
            return None, None, 0.0

        family = self._weak_prefix_to_family.get(prefix)
        if not family:
            return None, None, 0.0

        inferred_module = self._weak_leaf_hints.get(prefix) or family
        return family, inferred_module, 0.76

    def _strong_graph_family_vote(self, object_name: str) -> Tuple[Optional[str], int]:
        entry = self.objects.get(object_name)
        if not entry:
            return None, 0

        family_votes = Counter()
        for related_name in entry.get("approved_relations", []):
            related = self.objects.get(related_name)
            if not related:
                continue
            family = str(related.get("owning_module_family") or ModuleFamily.UNKNOWN.value)
            if family in {ModuleFamily.UNKNOWN.value, ModuleFamily.COMMON.value}:
                continue
            family_votes[family] += 1

        if not family_votes:
            return None, 0

        total = sum(family_votes.values())
        family, count = family_votes.most_common(1)[0]
        if total < 2 or (count / total) < 0.7:
            return None, total
        return family, total

    def _source_file_family_vote(self, object_name: str) -> Tuple[Optional[str], Optional[str], float]:
        entry = self.objects.get(object_name)
        if not entry:
            return None, None, 0.0

        family_votes = Counter()
        for source_name in entry.get("source_of_truth", []):
            family = SOURCE_FILE_FAMILY_HINTS.get(str(source_name))
            if family:
                family_votes[family] += 1

        if not family_votes:
            return None, None, 0.0

        total = sum(family_votes.values())
        family, count = family_votes.most_common(1)[0]
        if total > 1 and (count / total) < 0.7:
            return None, None, 0.0

        prefix = object_name.split("_", 1)[0]
        inferred_module = (
            self._seed_leaf_for_object(object_name)
            or self._weak_leaf_hints.get(prefix)
            or family
        )
        score = 0.78 if count >= 2 else 0.76
        return family, inferred_module, score

    def _apply_module_inference(self) -> None:
        # First pass: resolve from existing exact modules, taxonomy seeds, and SQL votes.
        for object_name, entry in self.objects.items():
            if entry.get("manual_lock"):
                entry["original_owning_module"] = entry.get("original_owning_module") or entry.get("owning_module", FusionModule.UNKNOWN.value)
                entry["low_confidence"] = bool(entry.get("confidence_score", 0.0) < MODULE_CONFIDENCE_THRESHOLD)
                continue

            original_module = str(entry.get("owning_module") or FusionModule.UNKNOWN.value)
            entry["original_owning_module"] = original_module

            if original_module != FusionModule.UNKNOWN.value:
                entry["owning_module_family"] = self._family_for_exact_module(original_module)
                entry["inferred_module"] = original_module
                entry["confidence_score"] = max(float(entry.get("confidence", 0.0)), 0.95)
                entry["inference_source"] = "metadata"
                entry["low_confidence"] = False
                continue

            seed_family = self._seed_family_for_object(object_name)
            seed_leaf = self._seed_leaf_for_object(object_name)
            sql_family, sql_leaf = self._strong_sql_family_vote(object_name)
            docs_family, docs_leaf, docs_score = self._strong_docs_family_vote(object_name)
            usage_family, usage_leaf, usage_score = self._strong_usage_family_vote(object_name)
            weak_family, weak_leaf, weak_score = self._weak_prefix_family_vote(object_name)
            graph_family, _ = self._strong_graph_family_vote(object_name)
            source_family, source_leaf, source_score = self._source_file_family_vote(object_name)

            if seed_family and any(
                corroborating_family == seed_family
                for corroborating_family in [usage_family, docs_family, sql_family, graph_family, source_family]
            ):
                inferred_module = seed_leaf or usage_leaf or docs_leaf or sql_leaf or source_leaf or seed_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = seed_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.95
                entry["inference_source"] = "composite"
                entry["low_confidence"] = False
                continue

            conflicting_families = {
                candidate
                for candidate in [usage_family, docs_family, sql_family, graph_family, source_family]
                if candidate and seed_family and candidate != seed_family
            }
            if seed_family and conflicting_families:
                inferred_module = seed_leaf or usage_leaf or docs_leaf or sql_leaf or source_leaf or seed_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = seed_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.6
                entry["inference_source"] = "conflict"
                entry["low_confidence"] = True
                continue

            if seed_family:
                inferred_module = seed_leaf or usage_leaf or docs_leaf or sql_leaf or seed_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = seed_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.9
                entry["inference_source"] = "taxonomy_sheet"
                entry["low_confidence"] = False
                continue

            if usage_family:
                inferred_module = usage_leaf or docs_leaf or sql_leaf or usage_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = usage_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = usage_score
                entry["inference_source"] = "validated_usage"
                entry["low_confidence"] = usage_score < MODULE_CONFIDENCE_THRESHOLD
                continue

            if docs_family:
                inferred_module = docs_leaf or sql_leaf or docs_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = docs_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = docs_score
                entry["inference_source"] = "docs"
                entry["low_confidence"] = docs_score < MODULE_CONFIDENCE_THRESHOLD
                continue

            if sql_family:
                inferred_module = sql_leaf or sql_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = sql_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.8
                entry["inference_source"] = "sql"
                entry["low_confidence"] = False
                continue

            if source_family and graph_family and source_family == graph_family:
                inferred_module = source_leaf or graph_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = source_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.82
                entry["inference_source"] = "relation_source"
                entry["low_confidence"] = False
                continue

            if source_family and source_score >= MODULE_CONFIDENCE_THRESHOLD:
                inferred_module = source_leaf or source_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = source_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = source_score
                entry["inference_source"] = "relation_source"
                entry["low_confidence"] = source_score < MODULE_CONFIDENCE_THRESHOLD
                continue

            if weak_family and weak_score >= MODULE_CONFIDENCE_THRESHOLD:
                inferred_module = weak_leaf or weak_family
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = weak_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = weak_score
                entry["inference_source"] = "weak_prefix"
                entry["low_confidence"] = weak_score < MODULE_CONFIDENCE_THRESHOLD
                continue

            entry["owning_module"] = FusionModule.UNKNOWN.value
            entry["owning_module_family"] = ModuleFamily.UNKNOWN.value
            entry["inferred_module"] = FusionModule.UNKNOWN.value
            entry["confidence_score"] = 0.0
            entry["inference_source"] = "none"
            entry["low_confidence"] = True

        # Second pass: iteratively propagate stable family assignments across the relationship graph.
        changed = True
        while changed:
            changed = False
            for object_name, entry in self.objects.items():
                if entry.get("manual_lock"):
                    continue
                if str(entry.get("owning_module_family")) != ModuleFamily.UNKNOWN.value:
                    continue

                graph_family, related_count = self._strong_graph_family_vote(object_name)
                if not graph_family or related_count < 2:
                    continue

                inferred_module = (
                    self._seed_leaf_for_object(object_name)
                    or self._weak_leaf_hints.get(object_name.split("_", 1)[0])
                    or graph_family
                )
                entry["owning_module"] = inferred_module
                entry["owning_module_family"] = graph_family
                entry["inferred_module"] = inferred_module
                entry["confidence_score"] = 0.78
                entry["inference_source"] = "graph"
                entry["low_confidence"] = entry["confidence_score"] < MODULE_CONFIDENCE_THRESHOLD
                changed = True

    def resolve_object_name(self, raw_name: str) -> Optional[str]:
        canonical = self._canonicalize(raw_name)
        if not canonical:
            return None
        if canonical in self.objects:
            return canonical
        return self.alias_to_object.get(canonical)

    def get_entry(self, raw_name: str) -> Optional[Dict[str, Any]]:
        canonical = self.resolve_object_name(raw_name)
        if canonical:
            return self.objects.get(canonical)
        return None

    def has_object(self, raw_name: str) -> bool:
        return self.resolve_object_name(raw_name) is not None

    def has_column(self, raw_table_name: str, column_name: str) -> bool:
        canonical = self.resolve_object_name(raw_table_name)
        if not canonical:
            return False
        return self._canonicalize_column(column_name) in self.columns_by_table.get(canonical, set())

    def get_related_objects(self, raw_name: str) -> List[str]:
        entry = self.get_entry(raw_name)
        if not entry:
            return []
        return sorted(set(entry.get("approved_relations", [])))

    def get_relation_details(self, raw_name: str, related_name: Optional[str] = None) -> List[Dict[str, Any]]:
        canonical = self.resolve_object_name(raw_name)
        if not canonical:
            return []
        if related_name is None:
            entry = self.objects.get(canonical) or {}
            return list(entry.get("relation_details", []))
        related_canonical = self.resolve_object_name(related_name)
        if not related_canonical:
            return []
        pair = tuple(sorted((canonical, related_canonical)))
        return list(self.relation_details_by_pair.get(pair, []))

    def get_primary_keys(self, raw_name: str) -> List[str]:
        entry = self.get_entry(raw_name)
        if not entry:
            return []
        return list(entry.get("primary_keys", []))

    def get_view_base_tables(self, raw_name: str) -> List[str]:
        entry = self.get_entry(raw_name)
        if not entry:
            return []
        return list(entry.get("base_tables", []))

    def find_join_path(self, raw_source: str, raw_target: str, max_depth: int = 4) -> List[str]:
        source = self.resolve_object_name(raw_source)
        target = self.resolve_object_name(raw_target)
        if not source or not target:
            return []
        if source == target:
            return [source]

        queue: deque[Tuple[str, List[str]]] = deque([(source, [source])])
        seen = {source}
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for neighbor in sorted(self.join_graph.get(current, set())):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return []

    def _query_tokens(self, query: str) -> Set[str]:
        tokens = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", query.upper()))
        lowered = query.lower()
        if "primary key" in lowered:
            tokens.add("PRIMARY_KEY")
        if "foreign key" in lowered or "join" in lowered:
            tokens.add("FOREIGN_KEY")
        return tokens

    def _score_entry(self, query: str, entry: Dict[str, Any], module: Optional[Any]) -> float:
        query_upper = query.upper()
        tokens = self._query_tokens(query)
        object_name = entry["object_name"]
        score = 0.0
        if object_name in query_upper:
            score += 100.0
        for alias in entry.get("aliases", []) + entry.get("ebs_aliases", []):
            if alias in query_upper:
                score += 60.0

        requested_families: Set[str] = set()
        if isinstance(module, (list, tuple, set)):
            for value in module:
                requested_families.update(module_families_for_value(value))
        elif module:
            requested_families.update(module_families_for_value(str(module)))

        if requested_families and entry.get("owning_module_family") in requested_families:
            score += 20.0

        name_tokens = set(object_name.split("_"))
        score += len(tokens.intersection(name_tokens)) * 5.0
        score += len(tokens.intersection(set(entry.get("columns", [])))) * 3.0
        if "FOREIGN_KEY" in tokens and entry.get("approved_relations"):
            score += 4.0
        if entry.get("manual_lock"):
            score += 1.5
        return score

    def build_schema_chunk(self, raw_name: str, max_columns: int = 8, max_relations: int = 5) -> Optional[Dict[str, Any]]:
        entry = self.get_entry(raw_name)
        if not entry:
            return None
        object_name = entry["object_name"]
        lines = [
            f"OBJECT: {object_name}",
            f"TYPE: {entry['object_type']}",
            f"MODULE_FAMILY: {entry.get('owning_module_family', ModuleFamily.UNKNOWN.value)}",
            f"INFERRED_MODULE: {entry.get('inferred_module', FusionModule.UNKNOWN.value)}",
        ]
        columns = entry.get("columns", [])[:max_columns]
        if columns:
            lines.append("COLUMNS: " + ", ".join(columns))
        relations = entry.get("approved_relations", [])[:max_relations]
        if relations:
            lines.append("RELATIONS: " + ", ".join(relations))
        primary_keys = entry.get("primary_keys", [])[:max_columns]
        if primary_keys:
            lines.append("PRIMARY_KEYS: " + ", ".join(primary_keys))
        base_tables = entry.get("base_tables", [])[:max_relations]
        if base_tables:
            lines.append("BASE_TABLES: " + ", ".join(base_tables))
        ebs_aliases = entry.get("ebs_aliases", [])[:3]
        if ebs_aliases:
            lines.append("EBS_ALIASES: " + ", ".join(ebs_aliases))
        if entry.get("manual_lock"):
            lines.append("MANUAL_LOCK: true")
        if entry.get("low_confidence"):
            lines.append(f"CONFIDENCE_SCORE: {entry.get('confidence_score', 0.0):.2f}")

        quality_score = max(float(entry.get("confidence", 0.0)), float(entry.get("confidence_score", 0.0)))
        return {
            "id": f"schema::{object_name}",
            "chunk_id": f"schema::{object_name}",
            "document_id": f"schema::{object_name}",
            "content": "\n".join(lines),
            "score": quality_score,
            "filename": f"schema_registry/{object_name}",
            "metadata": {
                "filename": f"schema_registry/{object_name}",
                "title": object_name,
                "module": entry.get("owning_module_family", ModuleFamily.UNKNOWN.value),
                "module_family": entry.get("owning_module_family", ModuleFamily.UNKNOWN.value),
                "inferred_module": entry.get("inferred_module", FusionModule.UNKNOWN.value),
                "owning_module": entry.get("owning_module", FusionModule.UNKNOWN.value),
                "manual_lock": bool(entry.get("manual_lock", False)),
                "primary_keys": list(entry.get("primary_keys", [])),
                "base_tables": list(entry.get("base_tables", [])),
                "task_type": "table_lookup",
                "doc_type": "schema_registry",
                "trusted_schema_objects": [object_name],
                "quality_score": quality_score,
                "source_system": "metadata",
                "source_uri": f"schema://{object_name}",
                "corpus": "schema_corpus",
                "content_hash": f"schema::{object_name}",
            },
        }

    def search(self, query: str, module: Optional[str] = None, limit: int = 4) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, str]] = []
        for object_name, entry in self.objects.items():
            score = self._score_entry(query, entry, module)
            if score > 0:
                scored.append((score, object_name))

        scored.sort(reverse=True)
        results: List[Dict[str, Any]] = []
        for score, object_name in scored[:limit]:
            chunk = self.build_schema_chunk(object_name)
            if not chunk:
                continue
            chunk["score"] = score
            chunk["combined_score"] = score
            results.append(chunk)
        return results

    def module_audit_report(self, baseline_unknown: Optional[int] = None) -> Dict[str, Any]:
        family_distribution = Counter(entry.get("owning_module_family", ModuleFamily.UNKNOWN.value) for entry in self.objects.values())
        unresolved_prefixes = Counter(
            name.split("_", 1)[0]
            for name, entry in self.objects.items()
            if entry.get("owning_module_family", ModuleFamily.UNKNOWN.value) == ModuleFamily.UNKNOWN.value
        )
        confidence_buckets = Counter()
        for entry in self.objects.values():
            score = float(entry.get("confidence_score", 0.0))
            if score >= 0.95:
                confidence_buckets[">=0.95"] += 1
            elif score >= 0.9:
                confidence_buckets["0.90-0.94"] += 1
            elif score >= MODULE_CONFIDENCE_THRESHOLD:
                confidence_buckets["0.75-0.89"] += 1
            else:
                confidence_buckets["<0.75"] += 1

        sample_names = [
            "XLA_AE_HEADERS",
            "FUN_USER_ROLE_DATA_ASGNMNTS",
            "IBY_VALIDATION_VALUES_",
            "MSC_SYSTEM_ITEMS",
            "EGO_ITEM_EFF_B",
            "IRC_MESSAGE_TRACKING",
            "WLF_LEARNING_ITEMS_F",
            "ANC_ABSENCE_PLANS_F",
            "PJB_ACCOUNTING_EVENTS",
            "PSC_INS_INSPECTION",
        ]
        samples = {name: self.objects.get(name) for name in sample_names if name in self.objects}
        report = {
            "total_objects": len(self.objects),
            "unknown_before": baseline_unknown,
            "unknown_after": family_distribution.get(ModuleFamily.UNKNOWN.value, 0),
            "unknown_reduction_pct": (
                round(((baseline_unknown - family_distribution.get(ModuleFamily.UNKNOWN.value, 0)) / baseline_unknown) * 100, 2)
                if baseline_unknown
                else None
            ),
            "family_distribution": dict(sorted(family_distribution.items())),
            "confidence_distribution": dict(confidence_buckets),
            "manual_locked_objects": sum(1 for entry in self.objects.values() if entry.get("manual_lock")),
            "low_confidence_objects": sum(1 for entry in self.objects.values() if entry.get("low_confidence")),
            "prefix_conflicts": self.taxonomy_conflicts.get("prefix_conflicts", {}),
            "top_unresolved_prefixes": unresolved_prefixes.most_common(20),
            "samples": samples,
        }
        return report


def get_default_registry() -> TrustedObjectRegistry:
    return TrustedObjectRegistry.get_default()
