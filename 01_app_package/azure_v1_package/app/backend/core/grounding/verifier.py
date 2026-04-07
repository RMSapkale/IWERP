import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
import structlog

from core.grounding.trusted_registry import get_default_registry
from core.schemas.router import FusionModule, ModuleFamily, TaskType, module_families_for_value

logger = structlog.get_logger(__name__)

INDEX_PATH = Path(__file__).parent / "fusion_tables_index.json"
METADATA_DIR = Path(__file__).parent / "metadata" / "csvs"
TABLE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$]{2,}$")
FAIL_CLOSED_MESSAGE = "Insufficient grounded data. Cannot generate verified answer."


class VerificationError(Exception):
    def __init__(self, message: str, retry_prompt: str):
        self.message = message
        self.retry_prompt = retry_prompt
        super().__init__(self.message)


class Verifier:
    """
    Multi-pass verifier for grounded Oracle Fusion outputs.
    """

    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(previous|all)\s+instructions",
        r"(?i)disregard\s+system\s+prompt",
        r"(?i)you\s+are\s+now\s+an?\s+expert",
    ]

    INTERNAL_MARKERS = [
        r"\[TERNARY_LOGIC\]",
        r"\[HIDDEN_REASONING_CHAIN\]",
        r"(?i)phase\s+\d+",
        r"<\|eot_id\|>",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
    ]

    SQL_REQUIRED_TASKS = {
        TaskType.SQL_GENERATION,
        TaskType.SQL_TROUBLESHOOTING,
        TaskType.REPORT_LOGIC,
    }
    FAST_FORMULA_REQUIRED_TASKS = {
        TaskType.FAST_FORMULA_GENERATION,
        TaskType.FAST_FORMULA_TROUBLESHOOTING,
    }
    DOC_CONTRACT_TASKS = {
        TaskType.PROCEDURE,
        TaskType.NAVIGATION,
        TaskType.TROUBLESHOOTING,
        TaskType.GENERAL,
        TaskType.SUMMARY,
        TaskType.INTEGRATION,
    }
    FORMULA_FUNCTIONS = {
        "ABS",
        "CEIL",
        "FLOOR",
        "ROUND",
        "TRUNC",
        "MOD",
        "POWER",
        "SQRT",
        "LENGTH",
        "SUBSTR",
        "INSTR",
        "TO_CHAR",
        "TO_DATE",
        "DAYS_BETWEEN",
        "MONTHS_BETWEEN",
        "ADD_DAYS",
        "ADD_MONTHS",
        "LEAST",
        "GREATEST",
        "NVL",
        "TO_NUMBER",
        "TO_TEXT",
        "WSA_GET",
        "WSA_SET",
        "PAY_INTERNAL_LOG_WRITE",
        "CHANGE_CONTEXTS",
    }
    FORMULA_RESERVED_WORDS = {
        "DEFAULT",
        "FOR",
        "IS",
        "INPUTS",
        "ARE",
        "IF",
        "THEN",
        "ELSE",
        "ENDIF",
        "END",
        "WHILE",
        "LOOP",
        "ENDLOOP",
        "RETURN",
        "AND",
        "OR",
        "NOT",
        "NULL",
        "WAS",
        "DEFAULTED",
    }

    def __init__(self, enable_rag: bool = True, enable_sql: bool = True, max_retries: int = 1):
        self.enable_rag = enable_rag
        self.enable_sql = enable_sql
        self.max_retries = max_retries
        self.registry = get_default_registry()
        self.global_tables: Set[str] = set(self.registry.objects.keys())
        self.table_columns: Dict[str, Set[str]] = {
            table: set(columns) for table, columns in self.registry.columns_by_table.items()
        }

        self.ebs_mapping = {}
        mapping_path = Path(__file__).parent / "fusion_ebs_mapping.json"
        if mapping_path.exists():
            try:
                with open(mapping_path, "r") as f:
                    self.ebs_mapping = json.load(f).get("mappings", {})
                logger.info("ebs_firewall_loaded", count=len(self.ebs_mapping))
            except Exception as exc:
                logger.error("ebs_firewall_load_failed", error=str(exc))

        self.index_registry = {
            "PO_HEADERS_ALL": ["SEGMENT1", "PO_HEADER_ID", "VENDOR_ID"],
            "AP_INVOICES_ALL": ["INVOICE_NUM", "INVOICE_ID", "VENDOR_ID", "ORG_ID"],
            "PER_ALL_PEOPLE_F": ["PERSON_ID", "PERSON_NUMBER"],
            "GL_JE_HEADERS": ["JE_HEADER_ID", "LEDGER_ID", "PERIOD_NAME"],
        }

        self.universal_bridge = {
            "ME21N": "PO_HEADERS_ALL",
            "MM01": "EGP_SYSTEM_ITEMS_B",
            "VA01": "DOO_HEADERS_ALL",
            "VENDOR_MASTER": "POZ_SUPPLIERS",
            "RCV_SHIPMENT_HEADERS": "CMR_RCV_TRANSACTIONS",
        }

    def _load_global_table_index(self) -> None:
        logger.info("verifier_index_loaded", count=len(self.global_tables))

    def _load_column_index(self) -> None:
        logger.info("verifier_columns_loaded", tables=len(self.table_columns))

    def _load_csv_columns(self, path: Path) -> None:
        with open(path, newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                table = str(row.get("TABLE_NAME") or "").upper()
                column = str(row.get("COLUMN_NAME") or "").upper()
                if TABLE_PATTERN.match(table) and TABLE_PATTERN.match(column):
                    self.table_columns.setdefault(table, set()).add(column)

    def _load_pipe_delimited_columns(self, path: Path) -> None:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            header = handle.readline().strip().lstrip("\ufeff").split("|")
            positions = {name: idx for idx, name in enumerate(header)}
            table_idx = positions.get("TABLE_NAME")
            column_idx = positions.get("COLUMN_NAME")
            if table_idx is None or column_idx is None:
                return
            for line in handle:
                parts = line.strip().split("|")
                if len(parts) <= max(table_idx, column_idx):
                    continue
                table = parts[table_idx].upper()
                column = parts[column_idx].upper()
                if TABLE_PATTERN.match(table) and TABLE_PATTERN.match(column):
                    self.table_columns.setdefault(table, set()).add(column)

    def verify_rag(self, answer: str, context_chunks: List[str]) -> Tuple[bool, Optional[str]]:
        if not self.enable_rag:
            return True, None

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, answer):
                return False, "Prompt injection detected in model output."

        citations = re.findall(r"\[D(\d+)\]", answer)
        if citations:
            max_cit = len(context_chunks)
            for citation in citations:
                if int(citation) > max_cit:
                    return False, f"Invalid citation [D{citation}]. Only grounded citations may be used."

        return True, None

    def _extract_sql_segments(self, output: str) -> List[str]:
        segments: List[str] = []

        def add_segment(segment: str) -> None:
            cleaned = segment.strip()
            if not cleaned:
                return
            if cleaned not in segments:
                segments.append(cleaned)

        for block in re.findall(r"```(?:sql)?\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL):
            add_segment(block)

        sql_section_match = re.search(
            r"\[SQL\](.*?)(?:\n\[[A-Za-z_]+\]|\Z)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if sql_section_match:
            add_segment(sql_section_match.group(1))

        for paragraph in re.split(r"\n\s*\n", output):
            candidate = paragraph.strip()
            candidate = re.sub(r"^\[SQL\]\s*", "", candidate, flags=re.IGNORECASE)
            if not re.match(r"(?is)^(with|select|insert|update|delete)\b", candidate):
                continue
            try:
                sqlglot.parse_one(candidate.strip().rstrip(";"), read="oracle")
                add_segment(candidate)
            except Exception:
                continue

        return segments

    def _collect_table_aliases(self, tree: exp.Expression) -> Tuple[Dict[str, str], Set[str], Set[str]]:
        alias_map: Dict[str, str] = {}
        derived_aliases: Set[str] = set()
        cte_aliases: Set[str] = set()

        for subquery in tree.find_all(exp.Subquery):
            if subquery.alias:
                derived_aliases.add(subquery.alias.upper())

        for cte in tree.find_all(exp.CTE):
            if cte.alias:
                cte_aliases.add(cte.alias.upper())

        for table in tree.find_all(exp.Table):
            table_name = table.name.upper()
            alias_name = (table.alias_or_name or table_name).upper()
            alias_map[alias_name] = table_name
            alias_map[table_name] = table_name

        return alias_map, derived_aliases, cte_aliases

    def _relation_column_for_table(self, table_name: str, column_name: str) -> str:
        canonical_table = self.registry.resolve_object_name(table_name.upper()) or table_name.upper()
        canonical_column = column_name.upper()
        if canonical_column != "ID":
            return canonical_column
        if self.registry.has_column(canonical_table, canonical_column):
            return canonical_column
        primary_keys = self.registry.get_primary_keys(canonical_table)
        if len(primary_keys) == 1 and self.registry.has_column(canonical_table, primary_keys[0]):
            return primary_keys[0].upper()
        return canonical_column

    def _validate_table_name(self, table_name: str) -> Tuple[bool, Optional[str]]:
        if table_name == "DUAL":
            return False, "Placeholder SQL using DUAL is not allowed."

        if table_name in self.ebs_mapping and str(self.ebs_mapping.get(table_name) or "").upper() != table_name:
            return False, (
                f"EBS legacy table '{table_name}' detected. Use the Fusion successor "
                f"'{self.ebs_mapping[table_name]}' instead."
            )

        if not self.registry.has_object(table_name):
            return False, f"Table '{table_name}' is not present in the Oracle Fusion metadata index."

        return True, None

    def _validate_columns(self, tree: exp.Expression, referenced_tables: Set[str]) -> Tuple[bool, Optional[str]]:
        alias_map, derived_aliases, cte_aliases = self._collect_table_aliases(tree)
        referenced_columns = {table: self.table_columns.get(table, set()) for table in referenced_tables}

        for column in tree.find_all(exp.Column):
            column_name = column.name.upper()
            if column_name == "*":
                return False, "SELECT * is not allowed. Enumerate grounded columns explicitly."

            qualifier = column.table.upper() if column.table else None
            if qualifier:
                if qualifier in derived_aliases or qualifier in cte_aliases:
                    continue
                target_table = alias_map.get(qualifier)
                if not target_table:
                    return False, f"Unresolved table alias '{qualifier}' in SQL."
                if not self.registry.has_column(target_table, column_name):
                    return False, f"Column '{column_name}' is not present on table '{target_table}'."
                continue

            if referenced_tables and not any(self.registry.has_column(table_name, column_name) for table_name in referenced_tables):
                return False, f"Column '{column_name}' is not present in the grounded table metadata."

        return True, None

    def _join_columns_match_relation(
        self,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> bool:
        details = self.registry.get_relation_details(left_table, right_table)
        if not details:
            return False
        for detail in details:
            source_table = str(detail.get("source_table") or "").upper()
            source_column = self._relation_column_for_table(
                source_table,
                str(detail.get("source_column") or "").upper(),
            )
            target_table = str(detail.get("target_table") or "").upper()
            target_column = self._relation_column_for_table(
                target_table,
                str(detail.get("target_column") or "").upper(),
            )
            if not source_column or not target_column:
                continue
            if (
                left_table == source_table
                and self._relation_column_for_table(left_table, left_column) == source_column
                and right_table == target_table
                and self._relation_column_for_table(right_table, right_column) == target_column
            ):
                return True
            if (
                left_table == target_table
                and self._relation_column_for_table(left_table, left_column) == target_column
                and right_table == source_table
                and self._relation_column_for_table(right_table, right_column) == source_column
            ):
                return True
        return False

    def _join_condition_references_multiple_tables(
        self,
        condition: exp.Expression,
        alias_map: Dict[str, str],
    ) -> bool:
        referenced = set()
        for column in condition.find_all(exp.Column):
            qualifier = column.table.upper() if column.table else ""
            if qualifier and qualifier in alias_map:
                referenced.add(alias_map[qualifier])
        return len(referenced) >= 2

    def verify_sql(self, sql: str, schema_context: str = "") -> Tuple[bool, Optional[str]]:
        if not self.enable_sql:
            return True, None

        sql_clean = sql.strip().strip(";").replace("`", '"')
        if not sql_clean:
            return False, "Empty SQL block detected."
        if re.search(r"(?m)--|/\*", sql_clean):
            return False, "SQL block must not contain inline or block comments."
        if re.search(r"(?i)\b(todo|placeholder|lorem ipsum|fake table)\b", sql_clean):
            return False, "SQL block contains placeholder or unfinished content."

        try:
            tree = sqlglot.parse_one(sql_clean, read="oracle")
        except Exception as exc:
            logger.warning("sql_syntax_error", error=str(exc), sql=sql_clean)
            return False, f"SQL syntax error: {exc}"

        if tree.find(exp.Star):
            return False, "SELECT * is not allowed. Enumerate grounded columns explicitly."

        if isinstance(tree, exp.Select) and not list(tree.find_all(exp.Table)):
            projections = list(tree.expressions)
            if projections and all(isinstance(expr, exp.Literal) for expr in projections):
                return False, "Literal-only SELECT statements are not allowed."
            return False, "SQL must reference grounded Oracle Fusion tables."

        referenced_tables: Set[str] = set()
        for table in tree.find_all(exp.Table):
            table_name = (self.registry.resolve_object_name(table.name.upper()) or table.name.upper())
            success, error_msg = self._validate_table_name(table_name)
            if not success:
                return False, error_msg
            referenced_tables.add(table_name)

        success, error_msg = self._validate_columns(tree, referenced_tables)
        if not success:
            return False, error_msg

        success, error_msg = self._validate_join_paths(tree, referenced_tables)
        if not success:
            return False, error_msg

        return True, None

    def verify_sql_style(self, sql: str) -> Tuple[bool, Optional[str]]:
        sql_clean = (sql or "").strip().strip(";")
        if not sql_clean:
            return False, "FAILED_SQL_STYLE_VIOLATION: empty SQL block detected."

        try:
            tree = sqlglot.parse_one(sql_clean, read="oracle")
        except Exception as exc:
            return False, f"FAILED_SQL_STYLE_VIOLATION: SQL syntax error: {exc}"

        tables = list(tree.find_all(exp.Table))
        for table in tables:
            alias = str(table.alias_or_name or "").upper()
            table_name = str(table.name or "").upper()
            if not alias or alias == table_name:
                return False, "FAILED_SQL_STYLE_VIOLATION: explicit table aliases are required."

        for column in tree.find_all(exp.Column):
            if column.name.upper() == "*":
                return False, "FAILED_SQL_STYLE_VIOLATION: SELECT * is not allowed."
            if not column.table:
                return False, "FAILED_SQL_STYLE_VIOLATION: alias-qualified columns are required."

        where_clause = tree.args.get("where")
        if where_clause is not None:
            for literal in where_clause.find_all(exp.Literal):
                literal_sql = literal.sql(dialect="oracle").strip().upper()
                if literal_sql in {"NULL"}:
                    continue
                return False, (
                    "FAILED_SQL_STYLE_VIOLATION: hardcoded filter literals are not allowed; "
                    "use bind placeholders."
                )

        return True, None

    def verify_sql_request_shape(self, sql: str, request_shape: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        if not request_shape:
            return True, None

        required_fields = list(request_shape.get("required_fields") or [])
        required_tables = {
            str(item).upper()
            for item in (request_shape.get("required_tables") or request_shape.get("required_entities") or [])
        }
        required_join_pairs_raw = list(request_shape.get("required_join_pairs") or [])
        required_alias_counts = {
            str(key).upper(): int(value)
            for key, value in (request_shape.get("required_table_alias_counts") or {}).items()
            if value
        }
        required_filters = list(request_shape.get("required_filters") or [])
        required_ordering = list(request_shape.get("required_ordering") or [])
        requires_join = bool(request_shape.get("needs_join") or request_shape.get("requires_join"))

        try:
            tree = sqlglot.parse_one((sql or "").strip().strip(";"), read="oracle")
        except Exception as exc:
            return False, f"FAILED_SQL_REQUEST_SHAPE_MISMATCH: SQL syntax error: {exc}"

        table_sequence: List[str] = []
        for table in tree.find_all(exp.Table):
            canonical = self.registry.resolve_object_name(table.name.upper()) or table.name.upper()
            table_sequence.append(canonical)
        table_set = set(table_sequence)
        table_counts = {table_name: table_sequence.count(table_name) for table_name in table_set}

        if required_tables and not required_tables.issubset(table_set):
            missing_tables = sorted(required_tables - table_set)
            return False, (
                "FAILED_SQL_REQUEST_SHAPE_MISMATCH: required business entities are missing: "
                + ", ".join(missing_tables)
            )

        for table_name, minimum_count in required_alias_counts.items():
            if table_counts.get(table_name, 0) < minimum_count:
                return False, (
                    "FAILED_SQL_REQUIRED_JOINS_MISSING: required table aliases or joins are missing for "
                    f"{table_name}."
                )

        if requires_join and len(table_sequence) < 2:
            return False, "FAILED_SQL_REQUIRED_JOINS_MISSING: multi-entity request requires grounded joins."

        if required_join_pairs_raw:
            normalized_pairs: List[Tuple[str, str]] = []
            for pair in required_join_pairs_raw:
                left = ""
                right = ""
                if isinstance(pair, str) and "->" in pair:
                    left, right = pair.split("->", 1)
                elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    left, right = str(pair[0]), str(pair[1])
                left = left.strip().upper()
                right = right.strip().upper()
                if left and right:
                    normalized_pairs.append((left, right))

            if normalized_pairs:
                adjacency = set()
                alias_map, _, _ = self._collect_table_aliases(tree)
                for join in tree.find_all(exp.Join):
                    target = join.this
                    target_table = ""
                    if isinstance(target, exp.Table):
                        target_table = (
                            self.registry.resolve_object_name(target.name.upper()) or target.name.upper()
                        )
                    on_clause = join.args.get("on")
                    if on_clause is not None:
                        for eq in on_clause.find_all(exp.EQ):
                            left = eq.left
                            right = eq.right
                            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                                continue
                            left_qualifier = left.table.upper() if left.table else ""
                            right_qualifier = right.table.upper() if right.table else ""
                            if not left_qualifier or not right_qualifier:
                                continue
                            left_table = alias_map.get(left_qualifier)
                            right_table = alias_map.get(right_qualifier)
                            if not left_table or not right_table or left_table == right_table:
                                continue
                            adjacency.add((left_table.upper(), right_table.upper()))
                            adjacency.add((right_table.upper(), left_table.upper()))
                    using_clause = join.args.get("using")
                    if using_clause is not None and target_table:
                        for table_name in table_set:
                            if table_name == target_table:
                                continue
                            adjacency.add((table_name.upper(), target_table.upper()))
                            adjacency.add((target_table.upper(), table_name.upper()))

                missing_pairs = [
                    f"{left}->{right}"
                    for left, right in normalized_pairs
                    if (left, right) not in adjacency
                ]
                if missing_pairs:
                    return False, (
                        "FAILED_SQL_REQUIRED_JOINS_MISSING: required join pairs are missing: "
                        + ", ".join(missing_pairs)
                    )

        select_expressions = list(getattr(tree, "expressions", []) or [])
        projection_columns: Set[str] = set()
        projection_aliases: Set[str] = set()
        for expression in select_expressions:
            alias_name = str(getattr(expression, "alias_or_name", "") or "").upper()
            if alias_name:
                projection_aliases.add(alias_name)
            for column in expression.find_all(exp.Column):
                projection_columns.add(column.name.upper())

        missing_fields: List[str] = []
        for field in required_fields:
            candidate_columns = {str(item).upper() for item in (field.get("columns") or [])}
            candidate_aliases = {str(item).upper() for item in (field.get("aliases") or [])}
            if candidate_columns & projection_columns:
                continue
            if candidate_aliases & projection_aliases:
                continue
            missing_fields.append(str(field.get("label") or field.get("key") or "field"))
        if missing_fields:
            return False, (
                "FAILED_SQL_REQUIRED_FIELDS_MISSING: requested fields are missing from the SELECT list: "
                + ", ".join(missing_fields)
            )

        where_clause = tree.args.get("where")
        where_columns: Set[str] = set()
        where_text = ""
        if where_clause is not None:
            where_text = where_clause.sql(dialect="oracle").upper()
            for column in where_clause.find_all(exp.Column):
                where_columns.add(column.name.upper())

        missing_filters: List[str] = []
        for filter_spec in required_filters:
            candidate_columns = {str(item).upper() for item in (filter_spec.get("columns") or [])}
            candidate_values = {str(item).upper() for item in (filter_spec.get("values") or [])}
            if candidate_columns & where_columns:
                if not candidate_values or any(value in where_text for value in candidate_values):
                    continue
            missing_filters.append(str(filter_spec.get("label") or filter_spec.get("key") or "filter"))
        if missing_filters:
            return False, (
                "FAILED_SQL_REQUEST_SHAPE_MISMATCH: required filter conditions are missing: "
                + ", ".join(missing_filters)
            )

        order_clause = tree.args.get("order")
        order_columns: Set[str] = set()
        order_text = ""
        if order_clause is not None:
            order_text = order_clause.sql(dialect="oracle").upper()
            for column in order_clause.find_all(exp.Column):
                order_columns.add(column.name.upper())

        missing_ordering: List[str] = []
        for ordering_spec in required_ordering:
            candidate_columns = {str(item).upper() for item in (ordering_spec.get("columns") or [])}
            candidate_aliases = {str(item).upper() for item in (ordering_spec.get("aliases") or [])}
            if candidate_columns & order_columns:
                continue
            if candidate_aliases and any(alias in order_text for alias in candidate_aliases):
                continue
            missing_ordering.append(str(ordering_spec.get("label") or ordering_spec.get("key") or "ordering"))
        if missing_ordering:
            return False, (
                "FAILED_SQL_REQUEST_SHAPE_MISMATCH: required ordering is missing: "
                + ", ".join(missing_ordering)
            )

        return True, None

    def _validate_join_paths(self, tree: exp.Expression, referenced_tables: Set[str]) -> Tuple[bool, Optional[str]]:
        if len(referenced_tables) <= 1:
            return True, None

        joins = list(tree.find_all(exp.Join))
        if not joins:
            return False, "Multi-table SQL must use explicit grounded JOIN clauses."

        alias_map, derived_aliases, cte_aliases = self._collect_table_aliases(tree)
        join_tables: List[str] = []
        for join in joins:
            target = join.this
            if isinstance(target, exp.Table):
                table_name = (self.registry.resolve_object_name(target.name.upper()) or target.name.upper())
                join_tables.append(table_name)
                if str(join.args.get("kind") or "").upper() == "CROSS":
                    return False, f"CROSS JOIN is not allowed for grounded Oracle Fusion SQL ({table_name})."
                if not join.args.get("on") and not join.args.get("using"):
                    return False, f"JOIN to '{table_name}' is missing an ON or USING clause."

        if not join_tables:
            return True, None

        connected_pairs = 0
        disconnected_tables: List[str] = []
        invalid_join_tables: List[str] = []
        for join in joins:
            target = join.this
            if not isinstance(target, exp.Table):
                continue
            table_name = (self.registry.resolve_object_name(target.name.upper()) or target.name.upper())
            related_tables = set(self.registry.get_related_objects(table_name))
            if related_tables & (referenced_tables - {table_name}):
                connected_pairs += 1
            else:
                disconnected_tables.append(table_name)
                continue

            join_validated = False
            using_clause = join.args.get("using")
            if using_clause is not None:
                using_columns: List[str] = []
                if isinstance(using_clause, list):
                    using_columns = [str(getattr(item, "name", item)).upper() for item in using_clause]
                else:
                    using_columns = [
                        str(getattr(item, "name", item)).upper()
                        for item in getattr(using_clause, "expressions", []) or []
                    ]
                for using_column in using_columns:
                    for related_table in related_tables & (referenced_tables - {table_name}):
                        if self._join_columns_match_relation(table_name, using_column, related_table, using_column):
                            join_validated = True
                            break
                    if join_validated:
                        break

            on_clause = join.args.get("on")
            if on_clause is not None:
                for eq in on_clause.find_all(exp.EQ):
                    left = eq.left
                    right = eq.right
                    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                        continue
                    left_qualifier = left.table.upper() if left.table else ""
                    right_qualifier = right.table.upper() if right.table else ""
                    if not left_qualifier or not right_qualifier:
                        continue
                    if left_qualifier in derived_aliases or right_qualifier in derived_aliases:
                        continue
                    if left_qualifier in cte_aliases or right_qualifier in cte_aliases:
                        continue
                    left_table = alias_map.get(left_qualifier)
                    right_table = alias_map.get(right_qualifier)
                    if not left_table or not right_table or left_table == right_table:
                        continue
                    left_column = left.name.upper()
                    right_column = right.name.upper()
                    if self._join_columns_match_relation(left_table, left_column, right_table, right_column):
                        join_validated = True
                        break
                    if (
                        left_column == right_column
                        and right_table in set(self.registry.get_related_objects(left_table))
                        and not self.registry.get_relation_details(left_table, right_table)
                    ):
                        join_validated = True
                        break
                if not join_validated and self._join_condition_references_multiple_tables(on_clause, alias_map):
                    join_validated = bool(related_tables & (referenced_tables - {table_name}))

            if not join_validated:
                invalid_join_tables.append(table_name)

        if disconnected_tables and connected_pairs == 0:
            return (
                False,
                "Join path is not grounded in the Oracle Fusion relationship metadata for: "
                + ", ".join(sorted(set(disconnected_tables))),
            )
        if invalid_join_tables:
            return (
                False,
                "Join predicates are not grounded in the Oracle Fusion relationship metadata for: "
                + ", ".join(sorted(set(invalid_join_tables))),
            )
        return True, None

    def _extract_formula_segments(self, output: str) -> List[str]:
        segments: List[str] = []

        def add_segment(segment: str) -> None:
            cleaned = (segment or "").strip()
            if cleaned and cleaned not in segments:
                segments.append(cleaned)

        formula_section_match = re.search(
            r"\[FORMULA\](.*?)(?:\n\[[A-Za-z_]+\]|\Z)",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if formula_section_match:
            add_segment(formula_section_match.group(1))

        for block in re.findall(r"```(?:text)?\s*(.*?)```", output, flags=re.IGNORECASE | re.DOTALL):
            if re.search(r"(?im)^\s*(inputs are|default for|return)\b", block):
                add_segment(block)

        return segments

    def _formula_type_semantics_match(self, formula: str, expected_formula_type: str) -> bool:
        expected = str(expected_formula_type or "").strip().lower()
        if not expected or expected in {"unknown", "n/a", "na", "none"}:
            return True

        lowered = (formula or "").lower()
        if "proration" in expected:
            return any(token in lowered for token in ("days_between", "calc_start_date", "calc_end_date", "proration"))
        if "validation" in expected or "eligibility" in expected:
            return any(token in lowered for token in ("entry_value", "validation", "valid", "invalid"))
        if "accrual" in expected or "absence" in expected:
            return any(token in lowered for token in ("months_between", "accrual", "eligible_flag", "term_date"))
        if "rate" in expected or "conversion" in expected:
            return any(token in lowered for token in ("conversion_rate", "base_amount", "round(", "rate"))
        if "skip" in expected:
            return "skip" in lowered or "return" in lowered
        if "extract" in expected:
            return any(token in lowered for token in ("to_char", "effective_date", "result_value"))
        if "time" in expected:
            return any(token in lowered for token in ("reported_hours", "hwm_", "date_earned"))
        if "payroll" in expected:
            return any(token in lowered for token in ("payroll", "result_value", "pay_"))
        return True

    def verify_fast_formula(
        self,
        formula: str,
        *,
        allowed_database_items: Optional[Iterable[str]] = None,
        allowed_contexts: Optional[Iterable[str]] = None,
        expected_formula_type: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        formula_clean = (formula or "").strip()
        if not formula_clean:
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Empty Fast Formula block detected."

        lowered = formula_clean.lower()
        if any(token in lowered for token in ["todo", "placeholder", "lorem ipsum"]):
            return False, "FAILED_FF_UNSUPPORTED_REQUEST: Placeholder or unfinished Fast Formula detected."
        if re.search(r"(?im)^\s*(select|with|insert|update|delete)\b", formula_clean):
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Fast Formula output contains SQL instead of Fast Formula syntax."
        if "return" not in lowered:
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Fast Formula must include a grounded RETURN statement."

        if_count = len(re.findall(r"(?im)^\s*if\b", formula_clean))
        endif_count = len(re.findall(r"(?im)\bendif\b|\bend\s+if\b", formula_clean))
        if if_count and endif_count < if_count:
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Fast Formula IF blocks are not properly closed with ENDIF."

        while_count = len(re.findall(r"(?im)^\s*while\b", formula_clean))
        endloop_count = len(re.findall(r"(?im)\bendloop\b|\bend\s+loop\b", formula_clean))
        if while_count and endloop_count < while_count:
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Fast Formula WHILE blocks are not properly closed with ENDLOOP."

        has_inputs = bool(re.search(r"(?im)^\s*inputs\s+are\b", formula_clean))
        has_defaults = bool(re.search(r"(?im)^\s*default\s+for\b", formula_clean))
        if not has_inputs and not has_defaults:
            return False, "FAILED_FF_STRUCTURE_MISMATCH: Fast Formula must include grounded INPUTS ARE or DEFAULT FOR statements."

        for line in formula_clean.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("DEFAULT FOR ") and " IS " not in stripped.upper():
                return False, "FAILED_FF_STRUCTURE_MISMATCH: DEFAULT FOR statements must include IS."

        suspicious_items = []
        for token in re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", formula_clean):
            if token in self.FORMULA_RESERVED_WORDS or token in self.FORMULA_FUNCTIONS:
                continue
            if token.startswith(("UNKNOWN_", "INVALID_", "MISSING_")):
                suspicious_items.append(token)
        if suspicious_items:
            return False, (
                "FAILED_FF_UNSUPPORTED_REQUEST: Unsupported Fast Formula database items or placeholders detected: "
                + ", ".join(sorted(set(suspicious_items)))
            )

        if allowed_contexts is not None:
            allowed_context_set = {str(item).upper() for item in allowed_contexts if str(item).strip()}
            if allowed_context_set:
                for match in re.finditer(r"(?is)change_contexts\s*\((.*?)\)", formula_clean):
                    context_clause = match.group(1)
                    referenced_contexts = {
                        token.upper()
                        for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", context_clause)
                        if token.upper() not in {"CHANGE_CONTEXTS", "AND", "OR", "NOT"}
                    }
                    unsupported_contexts = {
                        item
                        for item in referenced_contexts
                        if "_" in item and item not in allowed_context_set
                    }
                    if unsupported_contexts:
                        return False, (
                            "FAILED_FF_INVALID_CONTEXT: Unsupported Fast Formula contexts detected: "
                            + ", ".join(sorted(unsupported_contexts))
                        )

        if allowed_database_items is not None:
            allowed_dbi_set = {str(item).upper() for item in allowed_database_items if str(item).strip()}
            if allowed_dbi_set:
                unresolved_items = []
                for token in re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", formula_clean):
                    normalized = token.upper()
                    if normalized in self.FORMULA_RESERVED_WORDS or normalized in self.FORMULA_FUNCTIONS:
                        continue
                    if normalized in allowed_dbi_set:
                        continue
                    if "_" not in normalized:
                        continue
                    if re.match(r"^(FORMULA|INPUT|RESULT|RETURN)_", normalized):
                        continue
                    unresolved_items.append(normalized)
                if unresolved_items:
                    return False, (
                        "FAILED_FF_UNGROUNDED_VARIABLE: Formula references database items outside grounded allowlist: "
                        + ", ".join(sorted(set(unresolved_items))[:12])
                    )

        if expected_formula_type and not self._formula_type_semantics_match(formula_clean, expected_formula_type):
            return False, (
                "FAILED_FF_SEMANTIC_MISMATCH: Formula structure/content doesn't align with requested formula type '"
                + str(expected_formula_type)
                + "'."
            )

        return True, None

    def normalize_objects(self, text: str) -> Dict[str, List[str]]:
        potential_objects = set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", text))
        tags = {
            "confirmed_fusion": [],
            "mapped_from_ebs": [],
            "unconfirmed": [],
        }

        for obj in potential_objects:
            if self.registry.has_object(obj):
                tags["confirmed_fusion"].append(obj)
            elif obj in self.ebs_mapping:
                tags["mapped_from_ebs"].append(obj)
            else:
                tags["unconfirmed"].append(obj)

        return tags

    def verify_module_alignment(self, sql: str, module: Any) -> Tuple[bool, Optional[str]]:
        module_value = getattr(module, "value", module)
        if module_value in {FusionModule.UNKNOWN.value, FusionModule.COMMON.value, ModuleFamily.UNKNOWN.value}:
            return True, None

        try:
            tree = sqlglot.parse_one(sql.strip().strip(";"), read="oracle")
        except Exception:
            return False, "Unable to parse SQL for module alignment."

        allowed_families = module_families_for_value(str(module_value))
        allowed_families.discard(ModuleFamily.UNKNOWN.value)
        if not allowed_families:
            return True, None

        for table in tree.find_all(exp.Table):
            table_name = (self.registry.resolve_object_name(table.name.upper()) or table.name.upper())
            if table_name == "DUAL":
                return False, "Placeholder SQL using DUAL is not allowed."
            entry = self.registry.get_entry(table_name)
            if not entry:
                return False, f"Table '{table_name}' is not present in the Oracle Fusion metadata index."
            entry_family = str(entry.get("owning_module_family") or ModuleFamily.UNKNOWN.value)
            if entry_family in allowed_families:
                continue
            if entry_family == ModuleFamily.COMMON.value:
                continue
            return False, (
                f"Table '{table_name}' belongs to family '{entry_family}', "
                f"which does not align with the requested '{module_value}' module."
            )

        return True, None

    def verify_clean_output(self, answer: str) -> Tuple[bool, Optional[str]]:
        forbidden_phrases = [
            "i'm glad i could help",
            "as an ai",
            "i apologize",
            "sorry for",
            "let me know if you need anything else",
        ]
        answer_lower = answer.lower()
        for phrase in forbidden_phrases:
            if phrase in answer_lower:
                return False, f"Forbidden filler detected: '{phrase}'."

        for pattern in self.INTERNAL_MARKERS:
            if re.search(pattern, answer):
                return False, "Internal control text leaked into the final answer."

        return True, None

    def verify_performance_indexing(self, sql: str) -> Tuple[bool, Optional[str]]:
        try:
            tree = sqlglot.parse_one(sql.strip().strip(";"), read="oracle")
        except Exception:
            return False, "Unable to parse SQL for index checks."

        referenced_tables = [table.name.upper() for table in tree.find_all(exp.Table)]
        columns = [column.name.upper() for column in tree.find_all(exp.Column) if column.table]

        for table_name in referenced_tables:
            indexed_columns = self.index_registry.get(table_name)
            if not indexed_columns:
                continue
            for column_name in columns:
                if column_name not in indexed_columns:
                    logger.warning("performance_risk_detected", table=table_name, column=column_name)

        return True, None

    def verify_doc_answer_contract(self, task_type: TaskType, output: str) -> Tuple[bool, Optional[str]]:
        text = output or ""
        if not text.strip():
            return False, "FAILED_DOC_CONTRACT_EMPTY_OUTPUT: Output is empty."

        if task_type in {TaskType.PROCEDURE, TaskType.NAVIGATION}:
            if "Task:" not in text:
                return False, "FAILED_DOC_CONTRACT_PROCEDURE: missing Task section."
            if not re.search(r"(?m)^\s*Ordered Steps:\s*$", text):
                return False, "FAILED_DOC_CONTRACT_PROCEDURE: missing Ordered Steps section."
            if not re.search(r"(?m)^\s*1\.\s+", text):
                return False, "FAILED_DOC_CONTRACT_PROCEDURE: missing numbered steps."
            return True, None

        if task_type == TaskType.TROUBLESHOOTING:
            if not re.search(r"(?im)^\s*symptom:\s+", text):
                return False, "FAILED_DOC_CONTRACT_TROUBLESHOOTING: missing Symptom section."
            if not re.search(r"(?im)^\s*Likely Causes:\s*$", text):
                return False, "FAILED_DOC_CONTRACT_TROUBLESHOOTING: missing Likely Causes section."
            if not re.search(r"(?im)^\s*Resolution Steps:\s*$", text):
                return False, "FAILED_DOC_CONTRACT_TROUBLESHOOTING: missing Resolution Steps section."
            return True, None

        if task_type in {TaskType.GENERAL, TaskType.SUMMARY, TaskType.INTEGRATION}:
            if not re.search(r"(?im)^\s*Definition:\s+", text):
                return False, "FAILED_DOC_CONTRACT_SUMMARY: missing Definition section."
            if not re.search(r"(?im)^\s*Key Points:\s*$", text):
                return False, "FAILED_DOC_CONTRACT_SUMMARY: missing Key Points section."
            return True, None

        return True, None

    def run_pass(
        self,
        task_type: TaskType,
        output: str,
        context: List[Any],
        module: FusionModule = FusionModule.UNKNOWN,
        schema: str = "",
    ) -> Tuple[bool, str]:
        success, error_msg = self.verify_clean_output(output)
        if not success:
            return success, error_msg

        success, error_msg = self.verify_rag(output, [str(c) for c in context])
        if not success:
            return success, error_msg

        fail_closed = FAIL_CLOSED_MESSAGE in output

        if task_type in self.SQL_REQUIRED_TASKS:
            for marker in ("[MODULE]", "[GROUNDING]", "[SQL]", "[NOTES]"):
                if marker not in output:
                    return False, f"SQL lane output must include {marker}."

        if task_type in self.FAST_FORMULA_REQUIRED_TASKS:
            for marker in ("[FORMULA_TYPE]", "[GROUNDING]", "[FORMULA]", "[NOTES]"):
                if marker not in output:
                    return False, f"Fast Formula lane output must include {marker}."

        if fail_closed and task_type in (self.SQL_REQUIRED_TASKS | self.FAST_FORMULA_REQUIRED_TASKS):
            return True, ""

        if task_type in self.DOC_CONTRACT_TASKS and not fail_closed:
            success, error_msg = self.verify_doc_answer_contract(task_type, output)
            if not success:
                return success, error_msg

        sql_segments = self._extract_sql_segments(output)
        if task_type in self.SQL_REQUIRED_TASKS and not sql_segments:
            return False, "SQL is required for this task but no grounded SQL was found."

        for sql in sql_segments:
            success, error_msg = self.verify_sql(sql, schema)
            if not success:
                return success, error_msg

            success, error_msg = self.verify_sql_style(sql)
            if not success:
                return success, error_msg

            success, error_msg = self.verify_module_alignment(sql, module)
            if not success:
                return success, error_msg

            success, error_msg = self.verify_performance_indexing(sql)
            if not success:
                return success, error_msg

        formula_segments = self._extract_formula_segments(output)
        if task_type in self.FAST_FORMULA_REQUIRED_TASKS and not formula_segments:
            return False, "Fast Formula is required for this task but no grounded [FORMULA] block was found."

        for formula in formula_segments:
            success, error_msg = self.verify_fast_formula(formula)
            if not success:
                return success, error_msg

        return True, ""
