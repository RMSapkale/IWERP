import ast
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from core.grounding.verifier import Verifier
from core.ingest.curation import stable_hash
from core.ingest.specialization_tracks import (
    _flatten_section,
    _formula_type_from_text,
    _normalize_formula_record,
    _parse_formula_csv_blocks,
)


ROOT_DIR = Path("/Users/integrationwings/Desktop/LLM_Wrap")
BASE_DIR = ROOT_DIR / "iwerp-prod"
HARVEST_BASE_DIR = BASE_DIR / "specialization_tracks"
DEFAULT_SOURCE_REPO = "https://github.com/dipalibadgujar-integrationwings/Oraclewings_ai.git"
DEFAULT_SOURCE_BRANCH = "main"
DEFAULT_HARVEST_DIR = HARVEST_BASE_DIR / "oraclewings_ai_main_harvest"
MANIFEST_CLASSES = [
    "fast_formula_examples",
    "fast_formula_supporting_docs",
    "sql_runtime_assets",
    "plsql_knowledge_corpus",
    "hdl_fbdi_knowledge_corpus",
    "bip_otbi_knowledge_corpus",
    "agent_tool_rule_catalog",
    "reject_corpus",
]
SPARSE_PATHS = [
    "backend/backend_hcm",
    "backend/orawing_ai/agent",
    "backend/orawing_ai/api",
    "backend/orawing_ai/data",
    "backend/orawing_ai/fbdi",
    "backend/orawing_ai/llm",
    "backend/orawing_ai/validation",
    ".agent",
    ".agents",
]
FRONTEND_REJECT_PATTERNS = [
    re.compile(r"^frontend/", re.IGNORECASE),
    re.compile(r"^frontend/src/components/", re.IGNORECASE),
]
FAST_FORMULA_PATH_PATTERNS = [
    re.compile(r"fast.?formula", re.IGNORECASE),
    re.compile(r"formula_types", re.IGNORECASE),
    re.compile(r"formula_validator", re.IGNORECASE),
]
SQL_PATH_PATTERNS = [
    re.compile(r"(^|/)(sql|sql_)", re.IGNORECASE),
    re.compile(r"sql_generation", re.IGNORECASE),
    re.compile(r"sql_validator", re.IGNORECASE),
    re.compile(r"sql_filter", re.IGNORECASE),
    re.compile(r"\.sql$", re.IGNORECASE),
]
PLSQL_PATH_PATTERNS = [
    re.compile(r"plsql", re.IGNORECASE),
    re.compile(r"pl-sql", re.IGNORECASE),
    re.compile(r"procedure", re.IGNORECASE),
]
HDL_FBDI_PATH_PATTERNS = [
    re.compile(r"(^|/)(hdl|hsdl)(/|_|$)", re.IGNORECASE),
    re.compile(r"hdl_extract", re.IGNORECASE),
    re.compile(r"\bfbdi\b", re.IGNORECASE),
    re.compile(r"template", re.IGNORECASE),
]
BIP_OTBI_PATH_PATTERNS = [
    re.compile(r"(^|/|_)(bip|otbi)(/|_|$)", re.IGNORECASE),
    re.compile(r"\bbip\b", re.IGNORECASE),
    re.compile(r"\botbi\b", re.IGNORECASE),
    re.compile(r"xmlp", re.IGNORECASE),
    re.compile(r"bi_otbi", re.IGNORECASE),
]
AGENT_TOOL_RULE_PATH_PATTERNS = [
    re.compile(r"(^|/)\.agent(s)?/workflows/", re.IGNORECASE),
    re.compile(r"agent", re.IGNORECASE),
    re.compile(r"tool", re.IGNORECASE),
    re.compile(r"router", re.IGNORECASE),
    re.compile(r"validator", re.IGNORECASE),
    re.compile(r"prompt", re.IGNORECASE),
    re.compile(r"workflow", re.IGNORECASE),
    re.compile(r"orchestr", re.IGNORECASE),
    re.compile(r"planner", re.IGNORECASE),
    re.compile(r"knowledge_rules\.json$", re.IGNORECASE),
    re.compile(r"environment_rules\.json$", re.IGNORECASE),
]
MODULE_PATTERNS = [
    (re.compile(r"(?i)(hcm|payroll|absence|benefit|formula|recruit|workforce)"), "HCM"),
    (re.compile(r"(?i)(financial|finance|payables|receivables|ledger|asset|rmcs|tax|cash management)"), "Financials"),
    (re.compile(r"(?i)(procurement|supplier|purchasing|sourcing|purchase_order)"), "Procurement"),
    (re.compile(r"(?i)(scm|supply chain|inventory|manufacturing|shipping|order management)"), "SCM"),
    (re.compile(r"(?i)(project|ppm)"), "Projects"),
    (re.compile(r"(?i)(sales|crm|cx)"), "CRM"),
    (re.compile(r"(?i)(oic|integration|api|rest|soap)"), "Integration"),
]
CONTENT_TYPE_MAP = {
    ".py": "python",
    ".json": "json",
    ".txt": "text",
    ".md": "markdown",
    ".csv": "csv",
    ".sql": "sql",
    ".pdf": "pdf",
    ".docx": "docx",
    ".jsx": "jsx",
    ".html": "html",
    ".htm": "html",
    ".yaml": "yaml",
    ".yml": "yaml",
}
FORMULA_TYPE_KEYS = {"formula_type", "type_name", "name", "formulaType"}
MATERIALIZE_EXTENSIONS = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".csv",
    ".sql",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("_", " ").replace("-", " ")).strip() or "Untitled"


def _preview_text(value: str, limit: int = 1200) -> str:
    compact = re.sub(r"\s+", " ", (value or "").strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _content_type_for_path(path: Path) -> str:
    return CONTENT_TYPE_MAP.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "unknown")


def _infer_module(rel_path: str, text_hint: str = "") -> str:
    haystack = f"{rel_path} {text_hint}"
    for pattern, module in MODULE_PATTERNS:
        if pattern.search(haystack):
            return module
    return "Common"


def _infer_subsystem(rel_path: str) -> str:
    parts = Path(rel_path).parts
    if not parts:
        return "root"
    if parts[0].startswith(".agent"):
        return "/".join(parts[:2])
    if parts[0] == "backend" and len(parts) >= 3 and parts[1] == "orawing_ai":
        return "/".join(parts[:3])
    if parts[0] == "backend" and len(parts) >= 2:
        return "/".join(parts[:2])
    return "/".join(parts[: min(len(parts), 3)])


def _path_matches(rel_path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(rel_path) for pattern in patterns)


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    except Exception:
        return ""
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return unescape(text)


def _extract_pdf_text(path: Path) -> str:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(path))
            parts = []
            for page in reader.pages[:20]:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            continue
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(path), maxpages=20)
    except Exception:
        return ""


def _read_extractable_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _formula_types_from_path(path: Path) -> List[str]:
    if not path.exists():
        return []
    payload = _load_json(path)
    if isinstance(payload, list):
        return sorted({str(item).strip() for item in payload if str(item).strip()})
    if isinstance(payload, dict):
        values = []
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif isinstance(value, dict):
                for type_key in FORMULA_TYPE_KEYS:
                    maybe = value.get(type_key)
                    if isinstance(maybe, str) and maybe.strip():
                        values.append(maybe.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        values.append(item.strip())
                    elif isinstance(item, dict):
                        for type_key in FORMULA_TYPE_KEYS:
                            maybe = item.get(type_key)
                            if isinstance(maybe, str) and maybe.strip():
                                values.append(maybe.strip())
        return sorted(set(values))
    return []


def _looks_like_formula(text: str) -> bool:
    upper = (text or "").upper()
    if "RETURN" not in upper:
        return False
    return any(token in upper for token in ("INPUTS ARE", "DEFAULT FOR", "IF ", "ALIAS ", "FORMULA NAME"))


def _parse_formula_text_blocks(text: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    start_pattern = re.compile(r"^(?:\d+\.\d+\s+Template\s+—\s+.+FORMULA.*|✅\s*Example\s+\d+\s+—\s+.+)$", re.IGNORECASE)
    section_break_pattern = re.compile(r"^\d+\.\s+[A-Z].+$")

    def flush() -> None:
        nonlocal current_title, current_lines
        if not current_title:
            current_lines = []
            return
        content = "\n".join(line.rstrip() for line in current_lines).strip()
        if _looks_like_formula(content):
            blocks.append({"title": current_title, "content": content})
        current_title = None
        current_lines = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if start_pattern.match(stripped):
            flush()
            current_title = stripped
            continue
        if current_title and section_break_pattern.match(stripped) and not start_pattern.match(stripped):
            flush()
            continue
        if current_title is not None:
            current_lines.append(raw_line)
    flush()
    return blocks


def _extract_formula_candidates_from_json(payload: Any, trail: Optional[List[str]] = None) -> List[Dict[str, str]]:
    trail = trail or []
    candidates: List[Dict[str, str]] = []

    if isinstance(payload, dict):
        local_title = ""
        for key in ("title", "name", "formula_name", "use_case", "description"):
            maybe = payload.get(key)
            if isinstance(maybe, str) and maybe.strip():
                local_title = maybe.strip()
                break
        formula_value = None
        for key in ("formula", "content", "template", "example"):
            maybe = payload.get(key)
            if isinstance(maybe, str) and _looks_like_formula(maybe):
                formula_value = maybe
                break
        if formula_value:
            title = local_title or (trail[-1] if trail else "Formula Example")
            candidates.append({"title": _clean_title(title), "content": formula_value.strip()})
        for key, value in payload.items():
            next_trail = trail + [_clean_title(local_title or str(key))]
            if isinstance(value, str) and _looks_like_formula(value):
                title = local_title or str(key) or (trail[-1] if trail else "Formula Example")
                candidates.append({"title": _clean_title(title), "content": value.strip()})
            elif isinstance(value, (dict, list)):
                candidates.extend(_extract_formula_candidates_from_json(value, next_trail))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload, start=1):
            candidates.extend(_extract_formula_candidates_from_json(value, trail + [f"item {idx}"]))

    return candidates


def _python_asset_details(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    summary = ""
    entrypoint = ""
    dependencies: List[str] = []
    symbols: List[str] = []
    try:
        tree = ast.parse(text)
        summary = ast.get_docstring(tree) or ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        preferred = next((name for name in symbols if name in {"main", "run", "generate", "execute", "build"}), "")
        entrypoint = preferred or (symbols[0] if symbols else "")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
    except Exception:
        pass
    if not summary:
        summary = "\n".join(line.strip() for line in text.splitlines()[:8] if line.strip())
    dependencies = [item for item in dependencies if item][:12]
    return {
        "entrypoint": entrypoint,
        "dependencies_hint": sorted(dict.fromkeys(dependencies)),
        "extractable_text_summary": _preview_text(summary, limit=500),
        "top_level_symbols": symbols[:20],
    }


def _domain_tags(rel_path: str, text_hint: str = "") -> List[str]:
    lowered = f"{rel_path} {text_hint}".lower()
    tags = set()
    if any(token in lowered for token in ("sql", "select ", "join", "query", "bip sql")):
        tags.add("sql")
    if any(token in lowered for token in ("fast formula", "formula", "dbi", "database item")):
        tags.add("fast_formula")
    if any(token in lowered for token in ("hdl", "hsdl")):
        tags.add("hdl")
    if "fbdi" in lowered:
        tags.add("fbdi")
    if "plsql" in lowered or "pl/sql" in lowered:
        tags.add("plsql")
    if "bip" in lowered:
        tags.add("bip")
    if "otbi" in lowered:
        tags.add("otbi")
    if "hcm" in lowered:
        tags.add("hcm")
    if any(token in lowered for token in ("financial", "payables", "receivables", "ledger", "assets")):
        tags.add("financials")
    if any(token in lowered for token in ("procurement", "supplier", "purchasing")):
        tags.add("procurement")
    if any(token in lowered for token in ("scm", "supply chain", "inventory", "manufacturing")):
        tags.add("scm")
    return sorted(tags)


def _agent_asset_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    if "/workflows/" in lowered:
        return "workflow"
    if "validator" in lowered:
        return "validator"
    if "router" in lowered:
        return "router"
    if "prompt" in lowered:
        return "prompt"
    if "tool" in lowered:
        return "tool"
    if "agent" in lowered:
        return "agent"
    if "planner" in lowered:
        return "planner"
    if "orchestr" in lowered or "graph" in lowered:
        return "orchestrator"
    if lowered.endswith(".json"):
        return "rule_set"
    return "code_asset"


def _sql_asset_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    if lowered.endswith(".pdf"):
        return "guide"
    if lowered.endswith(".sql"):
        if "/schema/" in lowered:
            return "schema_sql"
        if "/migrations/" in lowered:
            return "migration_sql"
        return "sql_script"
    if "validator" in lowered:
        return "validator"
    if "tool" in lowered:
        return "tool"
    if "test" in lowered:
        return "test"
    if "node" in lowered:
        return "node"
    if "generation" in lowered or "generator" in lowered:
        return "generator"
    return "runtime_asset"


def _hdl_fbdi_asset_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    if lowered.endswith("fbdi_templates.json"):
        return "template_catalog"
    if "/mappings/" in lowered and lowered.endswith(".txt"):
        return "mapping_catalog"
    if lowered.endswith(".csv"):
        return "sample_upload"
    if lowered.endswith(".json"):
        return "knowledge_doc"
    if "service" in lowered:
        return "service_code"
    if "tool" in lowered:
        return "tool_code"
    if "endpoint" in lowered or "/api/" in lowered:
        return "api_code"
    return "support_code"


def _bip_asset_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    if lowered.endswith(".json"):
        return "knowledge_doc"
    if lowered.endswith(".txt") or lowered.endswith(".md") or lowered.endswith(".html"):
        return "reference_doc"
    return "support_asset"


def _plsql_asset_type(rel_path: str) -> str:
    lowered = rel_path.lower()
    if lowered.endswith(".json"):
        return "knowledge_doc"
    if lowered.endswith(".pdf"):
        return "guide"
    if lowered.endswith(".sql"):
        return "example_sql"
    return "support_asset"


def _extract_sql_examples_from_text(text: str) -> int:
    return len(re.findall(r"(?im)^\s*(SELECT|WITH)\b", text))


def _json_section_rows(payload: Any, title_prefix: str = "", max_sections: int = 50) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            section_title = _clean_title(f"{title_prefix} {key}".strip())
            flattened = _flatten_section(value)
            if flattened:
                rows.append((section_title, "\n".join(flattened[:120])))
            if len(rows) >= max_sections:
                break
    elif isinstance(payload, list):
        flattened = _flatten_section(payload)
        if flattened:
            rows.append((_clean_title(title_prefix or "List Section"), "\n".join(flattened[:120])))
    return rows[:max_sections]


def _parse_fbdi_mapping_text(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    chunk_pattern = re.compile(
        r"(?:Asset(?: name)?):\s*(?P<asset>.+?)\n"
        r"(?:Table(?: Name| name)):\s*(?P<table>.+?)\n"
        r"(?:File Link|Link):\s*(?P<link>.+?)(?:\n-+\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in chunk_pattern.finditer(text):
        asset_name = match.group("asset").strip()
        table_name = match.group("table").strip()
        link = match.group("link").strip()
        records.append(
            {
                "asset_name": asset_name,
                "table_name": table_name,
                "file_link": link,
            }
        )
    return records


@contextmanager
def _ephemeral_clone(source_repo: str, branch: str) -> Iterator[Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix=f"oraclewings_ai_{branch}_", dir="/tmp"))
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                "--branch",
                branch,
                source_repo,
                str(temp_dir),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class HarvestAccumulator:
    def __init__(self, *, source_repo: str, source_branch: str) -> None:
        self.source_repo = source_repo
        self.source_branch = source_branch
        self.records: Dict[str, List[Dict[str, Any]]] = {asset_class: [] for asset_class in MANIFEST_CLASSES}
        self.seen: Dict[str, set[str]] = {asset_class: set() for asset_class in MANIFEST_CLASSES}
        self.source_paths_by_class: Dict[str, set[str]] = {asset_class: set() for asset_class in MANIFEST_CLASSES}

    def add_record(
        self,
        *,
        asset_class: str,
        source_path: str,
        title: str,
        module: str,
        subsystem: str,
        content_type: str,
        normalized_payload: Dict[str, Any],
        confidence: float,
        dedupe_key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if asset_class not in self.records:
            raise ValueError(f"Unknown asset_class: {asset_class}")
        normalized = dict(normalized_payload)
        dedupe_key = dedupe_key or stable_hash(
            asset_class,
            title,
            json.dumps(normalized, sort_keys=True, ensure_ascii=True),
        )
        if dedupe_key in self.seen[asset_class]:
            self.reject(
                source_path=source_path,
                title=title,
                reason="duplicate_asset",
                normalized_payload={"dedupe_key": dedupe_key, "asset_class": asset_class},
                subsystem=subsystem,
                content_type=content_type,
            )
            return

        self.seen[asset_class].add(dedupe_key)
        record = {
            "asset_id": stable_hash(self.source_repo, self.source_branch, asset_class, dedupe_key)[:24],
            "asset_class": asset_class,
            "source_repo": self.source_repo,
            "source_branch": self.source_branch,
            "source_path": source_path,
            "title": title,
            "module": module,
            "subsystem": subsystem,
            "content_type": content_type,
            "normalized_payload": normalized,
            "confidence": round(float(confidence), 2),
            "dedupe_key": dedupe_key,
        }
        if extra:
            record.update(extra)
        self.records[asset_class].append(record)
        self.source_paths_by_class[asset_class].add(source_path)

    def reject(
        self,
        *,
        source_path: str,
        title: str,
        reason: str,
        normalized_payload: Optional[Dict[str, Any]] = None,
        subsystem: str = "",
        content_type: str = "unknown",
    ) -> None:
        payload = dict(normalized_payload or {})
        payload["reason"] = reason
        self.add_record(
            asset_class="reject_corpus",
            source_path=source_path,
            title=title or _clean_title(Path(source_path).name),
            module="Common",
            subsystem=subsystem or _infer_subsystem(source_path),
            content_type=content_type,
            normalized_payload=payload,
            confidence=0.0,
            dedupe_key=stable_hash("reject", source_path, title, reason, json.dumps(payload, sort_keys=True, ensure_ascii=True)),
        )


def _inventory_rel_paths(
    rel_paths: Iterable[str],
    *,
    source_repo: str,
    branch: str,
) -> Dict[str, Any]:
    rel_paths = sorted(rel_paths)
    extension_counts = Counter(Path(rel_path).suffix.lower() or "<no_ext>" for rel_path in rel_paths)
    top_level_counts = Counter(rel_path.split("/")[0] for rel_path in rel_paths)
    subsystem_counts = Counter(_infer_subsystem(rel_path) for rel_path in rel_paths)
    summary = {
        "generated_at": _iso_now(),
        "source_repo": source_repo,
        "source_branch": branch,
        "total_files": len(rel_paths),
        "extension_counts": dict(sorted(extension_counts.items(), key=lambda item: (-item[1], item[0]))),
        "top_level_counts": dict(sorted(top_level_counts.items(), key=lambda item: (-item[1], item[0]))),
        "subsystem_counts": dict(sorted(subsystem_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
    return summary


def _inventory_source_tree(
    source_dir: Path,
    *,
    source_repo: str,
    branch: str,
) -> Tuple[List[Path], Dict[str, Any]]:
    files = sorted(path for path in source_dir.rglob("*") if path.is_file() and ".git" not in path.parts)
    rel_paths = [path.relative_to(source_dir).as_posix() for path in files]
    summary = _inventory_rel_paths(rel_paths, source_repo=source_repo, branch=branch)
    return files, summary


def _git_list_repo_paths(repo_dir: Path) -> List[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _should_materialize_path(rel_path: str) -> bool:
    normalized = rel_path.replace("\\", "/")
    lowered = normalized.lower()
    if "/unwanted/" in lowered or lowered.startswith("unwanted/"):
        return False
    if not any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in SPARSE_PATHS):
        return False
    suffix = Path(normalized).suffix.lower()
    if suffix in MATERIALIZE_EXTENSIONS:
        pass
    elif not lowered.endswith(".jsonl"):
        return False
    if normalized.startswith(".agent/") or normalized.startswith(".agents/"):
        return True
    pattern_groups = (
        FAST_FORMULA_PATH_PATTERNS,
        SQL_PATH_PATTERNS,
        PLSQL_PATH_PATTERNS,
        HDL_FBDI_PATH_PATTERNS,
        BIP_OTBI_PATH_PATTERNS,
        AGENT_TOOL_RULE_PATH_PATTERNS,
    )
    return any(pattern.search(normalized) for patterns in pattern_groups for pattern in patterns)


def _materialize_repo_subset(repo_dir: Path, materialized_dir: Path, repo_paths: Iterable[str]) -> List[str]:
    materialized_paths: List[str] = []
    for rel_path in sorted(repo_paths):
        if not _should_materialize_path(rel_path):
            continue
        destination = materialized_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            subprocess.run(
                ["git", "-C", str(repo_dir), "show", f"HEAD:{rel_path}"],
                check=True,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        materialized_paths.append(rel_path)
    return materialized_paths


def _harvest_fast_formula_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    verifier = Verifier()
    formula_types_path = source_dir / "backend" / "backend_hcm" / "app" / "data" / "formula_types.json"
    known_formula_types = _formula_types_from_path(formula_types_path)

    csv_path = source_dir / "backend" / "backend_hcm" / "Fastformula_KT.csv"
    if csv_path.exists():
        for block in _parse_formula_csv_blocks(csv_path):
            valid, reason = verifier.verify_fast_formula(block["content"])
            rel_path = csv_path.relative_to(source_dir).as_posix()
            if not valid:
                accumulator.reject(
                    source_path=rel_path,
                    title=block["title"],
                    reason=reason or "formula_verifier_failed",
                    content_type="csv",
                )
                continue
            normalized = _normalize_formula_record(
                {
                    "source_path": rel_path,
                    "source_uri": rel_path,
                    "title": _clean_title(block["title"]),
                    "module": "HCM",
                    "task_type": "fast_formula_generation",
                    "doc_type": "fast_formula_example",
                    "content": block["content"],
                    "formula_type": _formula_type_from_text(f"{block['title']} {block['content']}", known_formula_types),
                    "use_case": _clean_title(block["title"]),
                },
                source_file=csv_path.name,
                source_sheet="csv_formula_blocks",
                confidence=0.92,
            )
            accumulator.add_record(
                asset_class="fast_formula_examples",
                source_path=rel_path,
                title=normalized["title"],
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="csv",
                normalized_payload={
                    "formula_name": normalized["formula_name"],
                    "formula_type": normalized["formula_type"],
                    "use_case": normalized["use_case"],
                    "inputs": normalized["input_values"],
                    "contexts": normalized["contexts"],
                    "database_items": normalized["database_items"],
                    "functions": normalized["functions"],
                    "return_behavior": normalized["return_behavior"],
                    "default_handling": normalized["default_handling"],
                    "formula_body": normalized["content"],
                    "source_kind": "csv_formula_block",
                    "source_file": normalized["source_file"],
                    "source_sheet": normalized["source_sheet"],
                },
                confidence=normalized["confidence"],
                dedupe_key=stable_hash(
                    "fast_formula_examples",
                    normalized["formula_name"],
                    normalized["formula_type"],
                    normalized["content"],
                ),
            )

    json_formula_paths = [
        source_dir / "backend" / "backend_hcm" / "app" / "data" / "fast_formula_knowledge.json",
        source_dir / "backend" / "orawing_ai" / "data" / "sources" / "hcm" / "FAST_FORMULA_KNOWLEDGE.json",
    ]
    for path in json_formula_paths:
        if not path.exists():
            continue
        rel_path = path.relative_to(source_dir).as_posix()
        payload = _load_json(path)
        for candidate in _extract_formula_candidates_from_json(payload):
            valid, reason = verifier.verify_fast_formula(candidate["content"])
            if not valid:
                accumulator.reject(
                    source_path=rel_path,
                    title=candidate["title"],
                    reason=reason or "formula_verifier_failed",
                    content_type="json",
                )
                continue
            normalized = _normalize_formula_record(
                {
                    "source_path": rel_path,
                    "source_uri": rel_path,
                    "title": candidate["title"],
                    "module": "HCM",
                    "task_type": "fast_formula_generation",
                    "doc_type": "fast_formula_example",
                    "content": candidate["content"],
                    "formula_type": _formula_type_from_text(f"{candidate['title']} {candidate['content']}", known_formula_types),
                    "use_case": candidate["title"],
                },
                source_file=path.name,
                source_sheet="json_formula_candidate",
                confidence=0.9,
            )
            accumulator.add_record(
                asset_class="fast_formula_examples",
                source_path=rel_path,
                title=normalized["title"],
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="json",
                normalized_payload={
                    "formula_name": normalized["formula_name"],
                    "formula_type": normalized["formula_type"],
                    "use_case": normalized["use_case"],
                    "inputs": normalized["input_values"],
                    "contexts": normalized["contexts"],
                    "database_items": normalized["database_items"],
                    "functions": normalized["functions"],
                    "return_behavior": normalized["return_behavior"],
                    "default_handling": normalized["default_handling"],
                    "formula_body": normalized["content"],
                    "source_kind": "json_formula_candidate",
                    "source_file": normalized["source_file"],
                    "source_sheet": normalized["source_sheet"],
                },
                confidence=normalized["confidence"],
                dedupe_key=stable_hash(
                    "fast_formula_examples",
                    normalized["formula_name"],
                    normalized["formula_type"],
                    normalized["content"],
                ),
            )
        for section_title, section_text in _json_section_rows(payload):
            if _looks_like_formula(section_text):
                continue
            accumulator.add_record(
                asset_class="fast_formula_supporting_docs",
                source_path=rel_path,
                title=section_title,
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="json",
                normalized_payload={
                    "source_kind": "json_support_section",
                    "section_title": section_title,
                    "content_preview": _preview_text(section_text, limit=1600),
                    "formula_type": _formula_type_from_text(section_title, known_formula_types),
                },
                confidence=0.84,
                dedupe_key=stable_hash("fast_formula_support", section_title, section_text),
            )

    if formula_types_path.exists():
        rel_path = formula_types_path.relative_to(source_dir).as_posix()
        for formula_type in known_formula_types:
            accumulator.add_record(
                asset_class="fast_formula_supporting_docs",
                source_path=rel_path,
                title=f"{formula_type} Formula Type",
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="json",
                normalized_payload={
                    "source_kind": "formula_type_catalog",
                    "formula_type": formula_type,
                    "content_preview": f"Oracle Fast Formula type: {formula_type}",
                },
                confidence=0.78,
                dedupe_key=stable_hash("formula_type", formula_type),
            )

    docx_path = source_dir / "backend" / "backend_hcm" / "Fast Formula Auto-Generation KT.docx"
    if docx_path.exists():
        rel_path = docx_path.relative_to(source_dir).as_posix()
        text = _extract_docx_text(docx_path)
        for block in _parse_formula_text_blocks(text):
            valid, reason = verifier.verify_fast_formula(block["content"])
            if not valid:
                accumulator.reject(
                    source_path=rel_path,
                    title=block["title"],
                    reason=reason or "formula_verifier_failed",
                    content_type="docx",
                )
                continue
            normalized = _normalize_formula_record(
                {
                    "source_path": rel_path,
                    "source_uri": rel_path,
                    "title": _clean_title(block["title"]),
                    "module": "HCM",
                    "task_type": "fast_formula_generation",
                    "doc_type": "fast_formula_example",
                    "content": block["content"],
                    "formula_type": _formula_type_from_text(f"{block['title']} {block['content']}", known_formula_types),
                    "use_case": _clean_title(block["title"]),
                },
                source_file=docx_path.name,
                source_sheet="docx_formula_blocks",
                confidence=0.88,
            )
            accumulator.add_record(
                asset_class="fast_formula_examples",
                source_path=rel_path,
                title=normalized["title"],
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="docx",
                normalized_payload={
                    "formula_name": normalized["formula_name"],
                    "formula_type": normalized["formula_type"],
                    "use_case": normalized["use_case"],
                    "inputs": normalized["input_values"],
                    "contexts": normalized["contexts"],
                    "database_items": normalized["database_items"],
                    "functions": normalized["functions"],
                    "return_behavior": normalized["return_behavior"],
                    "default_handling": normalized["default_handling"],
                    "formula_body": normalized["content"],
                    "source_kind": "docx_formula_block",
                    "source_file": normalized["source_file"],
                    "source_sheet": normalized["source_sheet"],
                },
                confidence=normalized["confidence"],
                dedupe_key=stable_hash(
                    "fast_formula_examples",
                    normalized["formula_name"],
                    normalized["formula_type"],
                    normalized["content"],
                ),
            )
        if text.strip():
            accumulator.add_record(
                asset_class="fast_formula_supporting_docs",
                source_path=rel_path,
                title=_clean_title(docx_path.stem),
                module="HCM",
                subsystem=_infer_subsystem(rel_path),
                content_type="docx",
                normalized_payload={
                    "source_kind": "docx_support_doc",
                    "content_preview": _preview_text(text, limit=1600),
                },
                confidence=0.8,
                dedupe_key=stable_hash("fast_formula_docx", _preview_text(text, limit=1600)),
            )

    for rel_path in (
        "backend/backend_hcm/app/validators/formula_validator.py",
        "backend/backend_hcm/merge_formula_types.py",
    ):
        path = source_dir / rel_path
        if not path.exists():
            continue
        details = _python_asset_details(path)
        accumulator.add_record(
            asset_class="fast_formula_supporting_docs",
            source_path=rel_path,
            title=_clean_title(path.stem),
            module="HCM",
            subsystem=_infer_subsystem(rel_path),
            content_type="python",
            normalized_payload={
                "source_kind": "formula_code_support",
                "entrypoint": details["entrypoint"],
                "dependencies_hint": details["dependencies_hint"],
                "extractable_text_summary": details["extractable_text_summary"],
                "top_level_symbols": details["top_level_symbols"],
            },
            confidence=0.82,
            dedupe_key=stable_hash("formula_support_code", rel_path),
        )


def _harvest_sql_runtime_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel_path = path.relative_to(source_dir).as_posix()
        if any(pattern.search(rel_path) for pattern in FRONTEND_REJECT_PATTERNS):
            continue
        if not _path_matches(rel_path, SQL_PATH_PATTERNS):
            continue
        content_type = _content_type_for_path(path)
        text = _read_extractable_text(path)
        module = _infer_module(rel_path, text[:400])
        payload = {
            "asset_type": _sql_asset_type(rel_path),
            "extractable_text_summary": _preview_text(text, limit=1600) if text else _clean_title(path.stem),
            "sql_example_count": _extract_sql_examples_from_text(text),
        }
        extra: Dict[str, Any] = {}
        if content_type == "python":
            details = _python_asset_details(path)
            payload.update(
                {
                    "entrypoint": details["entrypoint"],
                    "dependencies_hint": details["dependencies_hint"],
                    "top_level_symbols": details["top_level_symbols"],
                }
            )
        accumulator.add_record(
            asset_class="sql_runtime_assets",
            source_path=rel_path,
            title=_clean_title(path.stem),
            module=module,
            subsystem=_infer_subsystem(rel_path),
            content_type=content_type,
            normalized_payload=payload,
            confidence=0.86 if payload["asset_type"] != "guide" else 0.78,
            dedupe_key=stable_hash("sql_runtime", rel_path),
            extra=extra,
        )


def _harvest_plsql_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel_path = path.relative_to(source_dir).as_posix()
        if "unwanted/" in rel_path.lower():
            accumulator.reject(
                source_path=rel_path,
                title=_clean_title(path.stem),
                reason="unwanted_source_path",
                content_type=_content_type_for_path(path),
            )
            continue
        if not _path_matches(rel_path, PLSQL_PATH_PATTERNS):
            continue
        content_type = _content_type_for_path(path)
        if content_type == "json":
            payload = _load_json(path)
            for section_title, section_text in _json_section_rows(payload):
                accumulator.add_record(
                    asset_class="plsql_knowledge_corpus",
                    source_path=rel_path,
                    title=section_title,
                    module="Common",
                    subsystem=_infer_subsystem(rel_path),
                    content_type="json",
                    normalized_payload={
                        "asset_type": _plsql_asset_type(rel_path),
                        "content_preview": _preview_text(section_text, limit=1600),
                    },
                    confidence=0.86,
                    dedupe_key=stable_hash("plsql_section", section_title, section_text),
                )
        else:
            text = _read_extractable_text(path)
            accumulator.add_record(
                asset_class="plsql_knowledge_corpus",
                source_path=rel_path,
                title=_clean_title(path.stem),
                module="Common",
                subsystem=_infer_subsystem(rel_path),
                content_type=content_type,
                normalized_payload={
                    "asset_type": _plsql_asset_type(rel_path),
                    "content_preview": _preview_text(text, limit=1600) if text else _clean_title(path.name),
                },
                confidence=0.78 if content_type == "pdf" else 0.72,
                dedupe_key=stable_hash("plsql_asset", rel_path, _preview_text(text, limit=600)),
            )


def _harvest_hdl_fbdi_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel_path = path.relative_to(source_dir).as_posix()
        if any(pattern.search(rel_path) for pattern in FRONTEND_REJECT_PATTERNS):
            continue
        if not _path_matches(rel_path, HDL_FBDI_PATH_PATTERNS):
            continue
        content_type = _content_type_for_path(path)
        asset_type = _hdl_fbdi_asset_type(rel_path)
        module = _infer_module(rel_path)
        if rel_path.lower().endswith("hdl_extract_flow_knowledge.json"):
            payload = _load_json(path)
            for section_title, section_text in _json_section_rows(payload):
                accumulator.add_record(
                    asset_class="hdl_fbdi_knowledge_corpus",
                    source_path=rel_path,
                    title=section_title,
                    module="HCM",
                    subsystem=_infer_subsystem(rel_path),
                    content_type="json",
                    normalized_payload={
                        "asset_type": "knowledge_doc",
                        "content_preview": _preview_text(section_text, limit=1600),
                    },
                    confidence=0.88,
                    dedupe_key=stable_hash("hdl_section", section_title, section_text),
                )
            continue
        if rel_path.lower().endswith("fbdi_templates.json"):
            payload = _load_json(path)
            if isinstance(payload, dict):
                for template_key, template_meta in payload.items():
                    if not isinstance(template_meta, dict):
                        continue
                    accumulator.add_record(
                        asset_class="hdl_fbdi_knowledge_corpus",
                        source_path=rel_path,
                        title=_clean_title(str(template_meta.get("name") or template_key)),
                        module=_infer_module(rel_path, str(template_meta)),
                        subsystem=_infer_subsystem(rel_path),
                        content_type="json",
                        normalized_payload={
                            "asset_type": "template_catalog",
                            "oracle_job": template_meta.get("oracle_job"),
                            "template_link": template_meta.get("template_link"),
                            "files": template_meta.get("files", []),
                            "description": template_meta.get("description"),
                        },
                        confidence=0.91,
                        dedupe_key=stable_hash("fbdi_template", template_key, json.dumps(template_meta, sort_keys=True, ensure_ascii=True)),
                    )
            continue
        if "/mappings/" in rel_path.lower() and rel_path.lower().endswith(".txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            mapping_records = _parse_fbdi_mapping_text(text)
            for record in mapping_records:
                if record["table_name"].lower() == "not found" and record["file_link"].lower() == "not found":
                    accumulator.reject(
                        source_path=rel_path,
                        title=record["asset_name"],
                        reason="low_signal_mapping_entry",
                        normalized_payload=record,
                        content_type="text",
                    )
                    continue
                accumulator.add_record(
                    asset_class="hdl_fbdi_knowledge_corpus",
                    source_path=rel_path,
                    title=_clean_title(record["asset_name"]),
                    module=_infer_module(rel_path, record["asset_name"]),
                    subsystem=_infer_subsystem(rel_path),
                    content_type="text",
                    normalized_payload={
                        "asset_type": "mapping_catalog",
                        "table_name": record["table_name"],
                        "file_link": record["file_link"],
                    },
                    confidence=0.9,
                    dedupe_key=stable_hash("fbdi_mapping", record["asset_name"], record["table_name"], record["file_link"]),
                )
            continue
        text = _read_extractable_text(path)
        if rel_path.lower().endswith(".csv"):
            preview = "\n".join(text.splitlines()[:20])
        else:
            preview = _preview_text(text, limit=1600) if text else _clean_title(path.stem)
        payload = {
            "asset_type": asset_type,
            "content_preview": preview,
        }
        if content_type == "python":
            details = _python_asset_details(path)
            payload.update(
                {
                    "entrypoint": details["entrypoint"],
                    "dependencies_hint": details["dependencies_hint"],
                    "top_level_symbols": details["top_level_symbols"],
                }
            )
        accumulator.add_record(
            asset_class="hdl_fbdi_knowledge_corpus",
            source_path=rel_path,
            title=_clean_title(path.stem),
            module=module,
            subsystem=_infer_subsystem(rel_path),
            content_type=content_type,
            normalized_payload=payload,
            confidence=0.83 if asset_type.endswith("_code") else 0.8,
            dedupe_key=stable_hash("hdl_fbdi_asset", rel_path, preview),
        )


def _harvest_bip_otbi_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel_path = path.relative_to(source_dir).as_posix()
        if any(pattern.search(rel_path) for pattern in FRONTEND_REJECT_PATTERNS):
            continue
        if not _path_matches(rel_path, BIP_OTBI_PATH_PATTERNS):
            continue
        content_type = _content_type_for_path(path)
        if content_type == "json":
            payload = _load_json(path)
            for section_title, section_text in _json_section_rows(payload):
                accumulator.add_record(
                    asset_class="bip_otbi_knowledge_corpus",
                    source_path=rel_path,
                    title=section_title,
                    module=_infer_module(rel_path, section_title),
                    subsystem=_infer_subsystem(rel_path),
                    content_type="json",
                    normalized_payload={
                        "asset_type": _bip_asset_type(rel_path),
                        "content_preview": _preview_text(section_text, limit=1600),
                    },
                    confidence=0.84,
                    dedupe_key=stable_hash("bip_otbi_section", section_title, section_text),
                )
            continue
        text = _read_extractable_text(path)
        accumulator.add_record(
            asset_class="bip_otbi_knowledge_corpus",
            source_path=rel_path,
            title=_clean_title(path.stem),
            module=_infer_module(rel_path, text[:400]),
            subsystem=_infer_subsystem(rel_path),
            content_type=content_type,
            normalized_payload={
                "asset_type": _bip_asset_type(rel_path),
                "content_preview": _preview_text(text, limit=1600) if text else _clean_title(path.name),
            },
            confidence=0.78,
            dedupe_key=stable_hash("bip_otbi_asset", rel_path, _preview_text(text, limit=600)),
        )


def _harvest_agent_tool_rule_assets(source_dir: Path, accumulator: HarvestAccumulator) -> None:
    for path in sorted(p for p in source_dir.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel_path = path.relative_to(source_dir).as_posix()
        if any(pattern.search(rel_path) for pattern in FRONTEND_REJECT_PATTERNS):
            accumulator.reject(
                source_path=rel_path,
                title=_clean_title(path.stem),
                reason="frontend_only_component",
                content_type=_content_type_for_path(path),
            )
            continue
        if "unwanted/" in rel_path.lower():
            accumulator.reject(
                source_path=rel_path,
                title=_clean_title(path.stem),
                reason="unwanted_source_path",
                content_type=_content_type_for_path(path),
            )
            continue
        if not _path_matches(rel_path, AGENT_TOOL_RULE_PATH_PATTERNS):
            continue
        content_type = _content_type_for_path(path)
        text = _read_extractable_text(path)
        payload: Dict[str, Any] = {
            "asset_type": _agent_asset_type(rel_path),
            "domain_tags": _domain_tags(rel_path, text[:400]),
            "extractable_text_summary": _preview_text(text, limit=1600) if text else _clean_title(path.name),
            "sql_related": "sql" in rel_path.lower() or "sql" in text[:400].lower(),
            "formula_related": "formula" in rel_path.lower() or "fast formula" in text[:400].lower(),
            "hdl_related": any(token in rel_path.lower() for token in ("hdl", "fbdi", "hsdl")),
        }
        extra: Dict[str, Any] = {}
        if content_type == "python":
            details = _python_asset_details(path)
            payload.update(
                {
                    "entrypoint": details["entrypoint"],
                    "dependencies_hint": details["dependencies_hint"],
                }
            )
            payload["extractable_text_summary"] = details["extractable_text_summary"] or payload["extractable_text_summary"]
        elif content_type in {"markdown", "text", "json"}:
            payload.setdefault("entrypoint", "")
            payload.setdefault("dependencies_hint", [])
        accumulator.add_record(
            asset_class="agent_tool_rule_catalog",
            source_path=rel_path,
            title=_clean_title(path.stem),
            module=_infer_module(rel_path, text[:400]),
            subsystem=_infer_subsystem(rel_path),
            content_type=content_type,
            normalized_payload={
                "summary": payload["extractable_text_summary"],
            },
            confidence=0.81,
            dedupe_key=stable_hash("agent_tool_rule", rel_path),
            extra={
                "asset_type": payload["asset_type"],
                "domain_tags": payload["domain_tags"],
                "entrypoint": payload.get("entrypoint", ""),
                "dependencies_hint": payload.get("dependencies_hint", []),
                "extractable_text_summary": payload["extractable_text_summary"],
                "sql_related": payload["sql_related"],
                "formula_related": payload["formula_related"],
                "hdl_related": payload["hdl_related"],
            },
        )


def _harvest_decisions(source_repo: str, branch: str) -> Dict[str, Any]:
    return {
        "generated_at": _iso_now(),
        "source_repo": source_repo,
        "source_branch": branch,
        "storage_mode": "ephemeral_fetch",
        "scope": "extract_only",
        "no_push": True,
        "classification_rules": {
            "fast_formula_examples": [
                "Accept only formula bodies that pass the Fast Formula verifier.",
                "Deduplicate by formula_name + formula_type + formula_body.",
                "Parse examples from CSV, JSON formula nodes, and doc text blocks.",
            ],
            "fast_formula_supporting_docs": [
                "Keep formula knowledge sections, formula type catalogs, doc extracts, and validator/tooling code.",
                "Do not label support content as runnable examples.",
            ],
            "sql_runtime_assets": [
                "Keep SQL generators, validators, nodes, tools, tests, schema SQL, and guides as runtime assets.",
                "Do not promote runtime assets into grounded SQL examples during this harvest.",
            ],
            "plsql_knowledge_corpus": [
                "Keep PLSQL knowledge JSON and guide/doc assets as support knowledge only.",
            ],
            "hdl_fbdi_knowledge_corpus": [
                "Split mapping/template assets from support code and knowledge docs.",
                "Reject mapping entries with table/link both missing.",
            ],
            "bip_otbi_knowledge_corpus": [
                "Keep BIP/OTBI knowledge JSONs and crawled reference docs.",
            ],
            "agent_tool_rule_catalog": [
                "Catalog agents, tools, validators, routers, workflows, prompts, and rule JSONs.",
                "Frontend-only UI components are rejected rather than cataloged.",
            ],
            "reject_corpus": [
                "Reject duplicates, unwanted paths, low-signal mapping entries, and frontend-only UI components.",
            ],
        },
        "confidence_defaults": {
            "fast_formula_examples": 0.9,
            "fast_formula_supporting_docs": 0.82,
            "sql_runtime_assets": 0.86,
            "plsql_knowledge_corpus": 0.86,
            "hdl_fbdi_knowledge_corpus": 0.88,
            "bip_otbi_knowledge_corpus": 0.84,
            "agent_tool_rule_catalog": 0.81,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_outputs(
    output_dir: Path,
    accumulator: HarvestAccumulator,
    *,
    source_tree_summary: Dict[str, Any],
    decisions: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    sorted_records: Dict[str, List[Dict[str, Any]]] = {}
    for asset_class, rows in accumulator.records.items():
        ordered = sorted(rows, key=lambda row: (row["source_path"], row["title"], row["asset_id"]))
        sorted_records[asset_class] = ordered
        _write_jsonl(manifests_dir / f"{asset_class}.jsonl", ordered)

    reason_counts = Counter(
        row["normalized_payload"].get("reason", "unknown") for row in sorted_records["reject_corpus"]
    )
    rejection_report = {
        "generated_at": _iso_now(),
        "source_repo": accumulator.source_repo,
        "source_branch": accumulator.source_branch,
        "total_rejections": len(sorted_records["reject_corpus"]),
        "rejection_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "by_reason": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rejected_assets": sorted_records["reject_corpus"],
    }

    classified_paths = set()
    matched_paths_by_class: Dict[str, int] = {}
    for asset_class, paths in accumulator.source_paths_by_class.items():
        matched_paths_by_class[asset_class] = len(paths)
        classified_paths.update(paths)
    source_tree_summary = dict(source_tree_summary)
    source_tree_summary["matched_paths_by_class"] = matched_paths_by_class
    source_tree_summary["classified_file_count"] = len(classified_paths)
    source_tree_summary["unclassified_file_count"] = max(
        int(source_tree_summary.get("total_files", 0)) - len(classified_paths),
        0,
    )

    class_counts = {asset_class: len(rows) for asset_class, rows in sorted_records.items()}
    inventory_summary = {
        "generated_at": _iso_now(),
        "source_repo": accumulator.source_repo,
        "source_branch": accumulator.source_branch,
        "output_dir": str(output_dir),
        "class_counts": class_counts,
        "source_file_counts": {
            asset_class: len(paths) for asset_class, paths in accumulator.source_paths_by_class.items()
        },
        "total_manifest_rows": sum(class_counts.values()),
    }

    _write_json(output_dir / "inventory_summary.json", inventory_summary)
    _write_json(output_dir / "rejection_report.json", rejection_report)
    _write_json(output_dir / "source_tree_summary.json", source_tree_summary)
    _write_json(output_dir / "harvest_decisions.json", decisions)
    return inventory_summary


def _harvest_from_source_dir(
    source_dir: Path,
    *,
    output_dir: Path,
    source_repo: str,
    branch: str,
    source_tree_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if source_tree_summary is None:
        files, source_tree_summary = _inventory_source_tree(
            source_dir,
            source_repo=source_repo,
            branch=branch,
        )
        del files  # inventory is already captured in source_tree_summary
    accumulator = HarvestAccumulator(source_repo=source_repo, source_branch=branch)

    _harvest_fast_formula_assets(source_dir, accumulator)
    _harvest_sql_runtime_assets(source_dir, accumulator)
    _harvest_plsql_assets(source_dir, accumulator)
    _harvest_hdl_fbdi_assets(source_dir, accumulator)
    _harvest_bip_otbi_assets(source_dir, accumulator)
    _harvest_agent_tool_rule_assets(source_dir, accumulator)

    decisions = _harvest_decisions(source_repo, branch)
    return _write_outputs(
        output_dir,
        accumulator,
        source_tree_summary=source_tree_summary,
        decisions=decisions,
    )


def harvest_oraclewings_ai(
    *,
    source_repo: str = DEFAULT_SOURCE_REPO,
    branch: str = DEFAULT_SOURCE_BRANCH,
    output_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_output_dir = output_dir or (HARVEST_BASE_DIR / f"oraclewings_ai_{branch}_harvest")
    if source_dir is not None:
        return _harvest_from_source_dir(
            Path(source_dir),
            output_dir=resolved_output_dir,
            source_repo=source_repo,
            branch=branch,
        )
    with _ephemeral_clone(source_repo, branch) as cloned_dir:
        repo_paths = _git_list_repo_paths(cloned_dir)
        source_tree_summary = _inventory_rel_paths(
            repo_paths,
            source_repo=source_repo,
            branch=branch,
        )
        materialized_dir = cloned_dir / "__harvest_materialized__"
        materialized_dir.mkdir(parents=True, exist_ok=True)
        _materialize_repo_subset(cloned_dir, materialized_dir, repo_paths)
        return _harvest_from_source_dir(
            materialized_dir,
            output_dir=resolved_output_dir,
            source_repo=source_repo,
            branch=branch,
            source_tree_summary=source_tree_summary,
        )
