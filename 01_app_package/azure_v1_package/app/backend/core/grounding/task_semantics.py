import re
from typing import Any, Dict, List


class TaskSemanticAnalyzer:
    """
    Lightweight task-semantic matcher for task-shaped Oracle Fusion queries.
    It extracts canonical task signals from the query, annotates retrieved
    chunks with task evidence, and decides whether the retained grounding is
    semantically aligned enough to answer safely.
    """

    STRONG_MATCH_THRESHOLD = 0.5
    MEDIUM_MATCH_THRESHOLD = 0.25
    QUERY_SIGNAL_THRESHOLD = 0.5
    DOC_GROUNDING_CORPORA = {"docs_corpus", "troubleshooting_corpus"}
    HIGH_GROUNDING_SCORE = 0.8
    MEDIUM_GROUNDING_SCORE = 0.45
    BROAD_MODULE_NAMES = {"financials", "procurement", "hcm", "scm", "projects"}

    TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
        "invoice validation": {
            "patterns": [
                r"\binvoice validation\b",
                r"\bvalidate invoices?\b",
                r"\bvalidation\b.{0,24}\binvoice\b",
                r"\bautoinvoice\b.{0,24}\bvalidation\b",
                r"\bvalidation failures?\b",
                r"\buser-defined holds\b",
            ],
            "preferred_modules": ["Payables", "Receivables"],
            "correction": "Invoice validation is typically handled in Payables or Receivables invoice processing rather than {requested_module}.",
        },
        "payment terms": {
            "patterns": [
                r"\bpayment terms?\b",
                r"\bterms date\b",
                r"\bterms?\b.{0,20}\binstallments?\b",
                r"\binstallments?\b",
                r"\bpayment schedules?\b",
                r"\breference data sharing\b",
            ],
            "preferred_modules": ["Payables", "Receivables"],
            "correction": "Payment terms are typically configured in Payables or Receivables rather than {requested_module}.",
        },
        "supplier site setup": {
            "patterns": [
                r"\bsupplier site setup\b",
                r"\bsupplier sites?\b",
                r"\bsupplier onboarding\b",
                r"\bsupplier registration\b",
                r"\bonboard suppliers?\b",
            ],
            "preferred_modules": ["Payables", "Procurement"],
            "correction": "Supplier site setup is typically handled in supplier or procurement-related flows rather than {requested_module}.",
        },
        "journal approval": {
            "patterns": [
                r"\bjournal approval\b",
                r"\bapprove journals?\b",
                r"\bapproval group\b",
                r"\bjournal\b.{0,24}\bapproval\b",
                r"\binvoice approval\b",
                r"\bapproval workflow\b",
                r"\bapproval rules?\b",
                r"\binitiate\b.{0,24}\bapproval\b",
            ],
            "preferred_modules": ["Payables", "Receivables", "General Ledger"],
            "correction": "Journal approval is typically handled in journal or approval workflows rather than {requested_module}.",
        },
        "accounting period close": {
            "patterns": [
                r"\baccounting period close\b",
                r"\bperiod close\b",
                r"\bclose\b.{0,24}\bperiod\b",
                r"\bperiod close exceptions?\b",
                r"\bclosed period\b",
                r"\bclose the current\b.{0,32}\bperiod\b",
                r"\breceivables\s*-\s*period close\b",
            ],
            "preferred_modules": ["Payables", "Receivables", "General Ledger"],
            "correction": "Accounting period close is typically handled in subledger or ledger close flows rather than {requested_module}.",
        },
        "intercompany transaction": {
            "patterns": [
                r"\bintercompany transactions?\b",
                r"\bintercompany agreements?\b",
                r"\bmultitier intercompany\b",
                r"\bintercompany\b",
            ],
            "preferred_modules": ["Payables", "Receivables", "General Ledger", "Projects"],
            "correction": "Intercompany transactions are typically handled in intercompany or ledger flows rather than {requested_module}.",
        },
        "bank statement reconciliation": {
            "patterns": [
                r"\bbank statement reconciliation\b",
                r"\bautomatic reconciliation\b",
                r"\bmanual reconciliation\b",
                r"\breconciled groups?\b",
            ],
            "preferred_modules": ["Cash Management", "Receivables"],
            "correction": "Bank statement reconciliation is typically handled in Cash Management or reconciliation flows rather than {requested_module}.",
        },
        "asset capitalization": {
            "patterns": [
                r"\basset capitalization\b",
                r"\bcapitalize (an )?asset\b",
                r"\bcapitalization\b.{0,24}\basset\b",
            ],
            "preferred_modules": ["Assets"],
            "correction": "Asset capitalization is typically handled in Assets rather than {requested_module}.",
        },
        "expense report audit": {
            "patterns": [
                r"\bexpense report audit\b",
                r"\baudit expense report\b",
                r"\baudit list\b",
                r"\brequest more information\b",
                r"\brelease hold\b",
                r"\bwaive receipts?\b",
                r"\bconfirm manager approval\b",
            ],
            "preferred_modules": ["Expenses"],
            "correction": "Expense report audit is typically handled in Expenses rather than {requested_module}.",
        },
        "catalog upload": {
            "patterns": [
                r"\bcatalog upload\b",
                r"\bagreement loader\b",
                r"\bupload agreement lines?\b",
                r"\bupload file\b.{0,24}\bagreement\b",
                r"\bmap sets?\b",
            ],
            "preferred_modules": ["Procurement"],
            "correction": "Catalog upload processing is typically handled in Procurement loader flows rather than {requested_module}.",
        },
    }

    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").lower()).strip()

    @classmethod
    def _normalize_module(cls, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).lower()

    @classmethod
    def _title_and_content(cls, chunk: Dict[str, Any]) -> tuple[str, str]:
        metadata = chunk.get("metadata") or {}
        title = cls._normalize_text(
            str(
                metadata.get("title")
                or metadata.get("filename")
                or metadata.get("source_uri")
                or ""
            )
        )
        content = cls._normalize_text(
            str(
                chunk.get("content")
                or metadata.get("snippet")
                or metadata.get("summary")
                or ""
            )[:2000]
        )
        return title, content

    @classmethod
    def _score_text_for_task(cls, task_name: str, title: str, content: str) -> tuple[float, List[str]]:
        config = cls.TASK_CONFIGS[task_name]
        title_hits: List[str] = []
        content_hits: List[str] = []
        score = 0.0

        for pattern in config["patterns"]:
            title_match = re.search(pattern, title, flags=re.IGNORECASE)
            content_match = re.search(pattern, content, flags=re.IGNORECASE)
            if title_match:
                title_hits.append(title_match.group(0))
            if content_match:
                content_hits.append(content_match.group(0))

        if title_hits:
            score += 0.7
        if content_hits:
            score += 0.5

        # Boost exact canonical phrase matches.
        canonical = task_name.lower()
        if canonical in title:
            score += 0.2
        elif canonical in content:
            score += 0.12

        return min(score, 1.0), list(dict.fromkeys(title_hits + content_hits))

    @classmethod
    def extract_query_signals(cls, query: str) -> Dict[str, Any]:
        query_text = cls._normalize_text(query)
        signals: List[Dict[str, Any]] = []

        for task_name in cls.TASK_CONFIGS:
            score, matches = cls._score_text_for_task(task_name, query_text, query_text)
            if score < cls.QUERY_SIGNAL_THRESHOLD:
                continue
            signals.append(
                {
                    "task": task_name,
                    "confidence": round(score, 4),
                    "matches": matches,
                    "preferred_modules": list(cls.TASK_CONFIGS[task_name]["preferred_modules"]),
                }
            )

        signals.sort(key=lambda item: item["confidence"], reverse=True)
        top = signals[0] if signals else None
        return {
            "signals": signals,
            "top_task": top["task"] if top else None,
            "top_confidence": float(top["confidence"]) if top else 0.0,
        }

    @classmethod
    def annotate_chunks(cls, chunks: List[Dict[str, Any]], query_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        requested_tasks = [signal["task"] for signal in query_profile.get("signals", [])]

        for chunk in chunks:
            normalized = dict(chunk)
            metadata = dict(normalized.get("metadata") or {})
            title, content = cls._title_and_content(normalized)

            chunk_signals: List[str] = []
            per_task_scores: Dict[str, float] = {}
            matched_evidence: Dict[str, List[str]] = {}
            top_confidence = 0.0
            task_match_score = 0.0

            for task_name in cls.TASK_CONFIGS:
                score, matches = cls._score_text_for_task(task_name, title, content)
                if score <= 0.0:
                    continue
                per_task_scores[task_name] = round(score, 4)
                matched_evidence[task_name] = matches
                chunk_signals.append(task_name)
                top_confidence = max(top_confidence, score)
                if task_name in requested_tasks:
                    task_match_score = max(task_match_score, score)

            if requested_tasks and not task_match_score:
                for task_name in requested_tasks:
                    query_tokens = set(task_name.split())
                    token_hits = sum(1 for token in query_tokens if token in title or token in content)
                    if token_hits:
                        task_match_score = max(task_match_score, min(0.12 + (0.08 * token_hits), 0.28))

            if task_match_score >= cls.STRONG_MATCH_THRESHOLD:
                match_strength = "strong"
            elif task_match_score >= cls.MEDIUM_MATCH_THRESHOLD:
                match_strength = "medium"
            elif task_match_score > 0.0:
                match_strength = "weak"
            else:
                match_strength = "none"

            metadata["task_signals"] = chunk_signals
            metadata["task_confidence"] = round(top_confidence, 4)
            metadata["task_match_score"] = round(task_match_score, 4)
            metadata["task_match_strength"] = match_strength
            metadata["task_signal_evidence"] = matched_evidence
            normalized["metadata"] = metadata
            annotated.append(normalized)

        return annotated

    @classmethod
    def prioritize_chunks(cls, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(chunk: Dict[str, Any]) -> tuple[float, float, float]:
            metadata = chunk.get("metadata") or {}
            corpus = str(metadata.get("corpus") or "")
            corpus_boost = 1.0 if corpus in cls.DOC_GROUNDING_CORPORA else 0.4
            return (
                float(metadata.get("task_match_score") or 0.0),
                corpus_boost,
                float(chunk.get("combined_score") or chunk.get("score") or 0.0),
            )

        return sorted(chunks, key=sort_key, reverse=True)

    @classmethod
    def summarize_gate(
        cls,
        chunks: List[Dict[str, Any]],
        query_profile: Dict[str, Any],
        requested_module: str = "",
        module_explicit: bool = False,
    ) -> Dict[str, Any]:
        docs = [
            chunk
            for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") in cls.DOC_GROUNDING_CORPORA
        ]
        requested_norm = cls._normalize_module(requested_module)
        requested_exact_known = bool(requested_norm and requested_norm not in {"unknown"})

        def is_exact_module_doc(chunk: Dict[str, Any]) -> bool:
            if not requested_norm:
                return False
            metadata = chunk.get("metadata") or {}
            actual_norm = cls._normalize_module(str(metadata.get("module") or ""))
            return bool(actual_norm and actual_norm == requested_norm)

        strong_docs = [
            chunk for chunk in docs
            if (chunk.get("metadata") or {}).get("task_match_strength") == "strong"
        ]
        medium_docs = [
            chunk for chunk in docs
            if (chunk.get("metadata") or {}).get("task_match_strength") == "medium"
        ]
        exact_docs = [chunk for chunk in docs if is_exact_module_doc(chunk)]
        exact_strong_docs = [chunk for chunk in strong_docs if is_exact_module_doc(chunk)]
        exact_medium_docs = [chunk for chunk in medium_docs if is_exact_module_doc(chunk)]
        sibling_strong_docs = [chunk for chunk in strong_docs if not is_exact_module_doc(chunk)]
        sibling_medium_docs = [chunk for chunk in medium_docs if not is_exact_module_doc(chunk)]

        top_task = query_profile.get("top_task")
        preferred_modules = set(
            cls._normalize_module(module_name)
            for module_name in cls.TASK_CONFIGS.get(str(top_task or ""), {}).get("preferred_modules", [])
        )
        module_correction = None
        gate_reason = "no_query_signal"
        module_conflict = False
        grounding_score = 0.0
        decision_score = 0.0
        grounding_tier = "low"
        decision_tier = "low"

        if query_profile.get("signals"):
            requested_is_broad_family = requested_norm in cls.BROAD_MODULE_NAMES
            if (
                module_explicit
                and requested_exact_known
                and preferred_modules
                and requested_norm not in preferred_modules
                and not requested_is_broad_family
            ):
                module_conflict = True
                module_correction = cls.module_correction_message(query_profile, requested_module)
                gate = "FAILED"
                gate_reason = "requested_module_not_preferred_for_task"
            elif exact_strong_docs:
                gate = "PASSED"
                gate_reason = "exact_module_strong_match"
            elif exact_medium_docs:
                gate = "PASSED"
                gate_reason = "exact_module_medium_match"
            elif module_explicit and requested_exact_known and exact_docs:
                gate = "PASSED"
                gate_reason = "exact_module_weak_match"
            elif not module_explicit and strong_docs:
                gate = "PASSED"
                gate_reason = "family_or_ambiguous_strong_match"
            elif not module_explicit and medium_docs:
                gate = "PASSED"
                gate_reason = "family_or_ambiguous_medium_match"
            elif (
                module_explicit
                and requested_exact_known
                and requested_norm in preferred_modules
                and (sibling_strong_docs or sibling_medium_docs)
            ):
                gate = "PASSED"
                gate_reason = "preferred_module_family_match"
            elif (
                module_explicit
                and requested_exact_known
                and requested_norm
                and not exact_docs
                and (sibling_strong_docs or sibling_medium_docs)
            ):
                module_correction = cls.module_correction_message(query_profile, requested_module)
                if not module_correction:
                    sibling_modules = sorted(
                        {
                            str((chunk.get("metadata") or {}).get("module") or "").strip()
                            for chunk in (sibling_strong_docs or sibling_medium_docs)
                            if str((chunk.get("metadata") or {}).get("module") or "").strip()
                        }
                    )
                    if sibling_modules:
                        sibling_label = " or ".join(sibling_modules[:2])
                        module_correction = (
                            f'The retained grounding aligns to {sibling_label} rather than {requested_module}.'
                        )
                gate = "FAILED"
                gate_reason = "sibling_module_only_match"
            elif strong_docs:
                gate = "PASSED"
                gate_reason = "strong_match_without_explicit_module"
            else:
                gate = "FAILED"
                gate_reason = "no_strong_or_medium_match"
        else:
            gate = "PASSED"

        grounding_score = min(
            1.0,
            (0.95 if exact_strong_docs else 0.0)
            + (0.72 if exact_medium_docs else 0.0)
            + (
                0.36
                if module_explicit and requested_exact_known and exact_docs and not (exact_strong_docs or exact_medium_docs)
                else 0.0
            )
            + min(0.45, len(strong_docs) * 0.18)
            + min(0.24, len(medium_docs) * 0.08),
        )
        if grounding_score >= cls.HIGH_GROUNDING_SCORE:
            grounding_tier = "high"
        elif grounding_score >= cls.MEDIUM_GROUNDING_SCORE:
            grounding_tier = "medium"

        decision_score = min(
            1.0,
            (float(query_profile.get("top_confidence") or 0.0) * 0.45)
            + (grounding_score * 0.45)
            + (0.1 if not module_conflict else 0.0),
        )
        if decision_score >= 0.8:
            decision_tier = "high"
        elif decision_score >= 0.5:
            decision_tier = "medium"

        return {
            "requested_task_signals": query_profile.get("signals", []),
            "top_task_signal": top_task,
            "top_task_confidence": query_profile.get("top_confidence", 0.0),
            "docs_count": len(docs),
            "strong_doc_count": len(strong_docs),
            "medium_doc_count": len(medium_docs),
            "exact_doc_count": len(exact_docs),
            "exact_strong_doc_count": len(exact_strong_docs),
            "exact_medium_doc_count": len(exact_medium_docs),
            "sibling_strong_doc_count": len(sibling_strong_docs),
            "sibling_medium_doc_count": len(sibling_medium_docs),
            "task_match_rate": round(len(strong_docs) / max(len(docs), 1), 4) if docs else 0.0,
            "exact_task_match_rate": round(len(exact_strong_docs) / max(len(exact_docs), 1), 4) if exact_docs else 0.0,
            "task_semantic_gate": gate,
            "task_gate_reason": gate_reason,
            "module_correction_message": module_correction,
            "module_conflict": module_conflict,
            "preferred_modules": sorted(preferred_modules),
            "grounding_evidence_score": round(grounding_score, 4),
            "grounding_confidence_tier": grounding_tier,
            "decision_confidence_score": round(decision_score, 4),
            "decision_confidence_tier": decision_tier,
        }

    @classmethod
    def filter_prompt_chunks(
        cls,
        chunks: List[Dict[str, Any]],
        query_profile: Dict[str, Any],
        docs_expected: bool,
        requested_module: str = "",
        module_explicit: bool = False,
        preferred_module_allowlist: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if not docs_expected or not query_profile.get("signals"):
            return chunks

        requested_norm = cls._normalize_module(requested_module)
        requested_exact_known = bool(requested_norm and requested_norm not in {"unknown"})
        preferred_allowlist = {
            cls._normalize_module(module_name)
            for module_name in (preferred_module_allowlist or [])
            if cls._normalize_module(module_name)
        }
        top_signal = (query_profile.get("signals") or [{}])[0] if query_profile.get("signals") else {}
        task_preferred_modules = {
            cls._normalize_module(module_name)
            for module_name in top_signal.get("preferred_modules", [])
            if cls._normalize_module(module_name)
        }

        def is_exact_module_doc(chunk: Dict[str, Any]) -> bool:
            if not requested_norm:
                return False
            metadata = chunk.get("metadata") or {}
            actual_norm = cls._normalize_module(str(metadata.get("module") or ""))
            return bool(actual_norm and actual_norm == requested_norm)

        def is_preferred_module_doc(chunk: Dict[str, Any]) -> bool:
            if not preferred_allowlist:
                return False
            metadata = chunk.get("metadata") or {}
            actual_norm = cls._normalize_module(str(metadata.get("module") or ""))
            return bool(actual_norm and actual_norm in preferred_allowlist)

        strong_docs = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") in cls.DOC_GROUNDING_CORPORA
            and (chunk.get("metadata") or {}).get("task_match_strength") == "strong"
        ]
        medium_docs = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") in cls.DOC_GROUNDING_CORPORA
            and (chunk.get("metadata") or {}).get("task_match_strength") == "medium"
        ]
        all_docs = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") in cls.DOC_GROUNDING_CORPORA
        ]
        non_docs = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("corpus") not in cls.DOC_GROUNDING_CORPORA
        ]
        exact_docs = [chunk for chunk in all_docs if is_exact_module_doc(chunk)]
        exact_strong_docs = [chunk for chunk in strong_docs if is_exact_module_doc(chunk)]
        exact_medium_docs = [chunk for chunk in medium_docs if is_exact_module_doc(chunk)]
        preferred_docs = [chunk for chunk in all_docs if is_preferred_module_doc(chunk)]
        preferred_strong_docs = [chunk for chunk in strong_docs if is_preferred_module_doc(chunk)]
        preferred_medium_docs = [chunk for chunk in medium_docs if is_preferred_module_doc(chunk)]
        task_preferred_docs = [
            chunk
            for chunk in all_docs
            if cls._normalize_module(str((chunk.get("metadata") or {}).get("module") or "")) in task_preferred_modules
        ]
        task_preferred_strong_docs = [
            chunk
            for chunk in strong_docs
            if cls._normalize_module(str((chunk.get("metadata") or {}).get("module") or "")) in task_preferred_modules
        ]
        task_preferred_medium_docs = [
            chunk
            for chunk in medium_docs
            if cls._normalize_module(str((chunk.get("metadata") or {}).get("module") or "")) in task_preferred_modules
        ]

        if module_explicit and requested_exact_known and exact_strong_docs:
            return exact_strong_docs[:2] + non_docs[:1]
        if module_explicit and requested_exact_known and exact_medium_docs:
            return exact_medium_docs[:2] + non_docs[:1]
        if module_explicit and requested_exact_known and exact_docs:
            return exact_docs[:2] + non_docs[:1]
        if module_explicit and requested_exact_known and requested_norm in task_preferred_modules:
            if task_preferred_strong_docs:
                return task_preferred_strong_docs[:2] + non_docs[:1]
            if task_preferred_medium_docs:
                return task_preferred_medium_docs[:2] + non_docs[:1]
            if task_preferred_docs:
                return task_preferred_docs[:2] + non_docs[:1]
        if module_explicit and preferred_strong_docs:
            return preferred_strong_docs[:2] + non_docs[:1]
        if module_explicit and preferred_medium_docs:
            return preferred_medium_docs[:2] + non_docs[:1]
        if module_explicit and preferred_docs:
            return preferred_docs[:2] + non_docs[:1]

        if not module_explicit and requested_norm in {"", "unknown"}:
            if task_preferred_strong_docs:
                return task_preferred_strong_docs[:2] + non_docs[:1]
            if task_preferred_medium_docs:
                return task_preferred_medium_docs[:2] + non_docs[:1]
            if task_preferred_docs:
                return task_preferred_docs[:2] + non_docs[:1]
            if all_docs:
                return all_docs[:1] + non_docs[:1]

        if strong_docs:
            return strong_docs + non_docs[:1]
        if medium_docs:
            return medium_docs[:1] + non_docs[:1]
        return non_docs[:1]

    @classmethod
    def module_correction_message(cls, query_profile: Dict[str, Any], requested_module: str) -> str | None:
        top_task = query_profile.get("top_task")
        top_confidence = float(query_profile.get("top_confidence") or 0.0)
        requested = (requested_module or "").strip()
        if not top_task or not requested or top_confidence < 0.6:
            return None

        preferred_modules = set(cls.TASK_CONFIGS[top_task]["preferred_modules"])
        if requested in preferred_modules:
            return None
        return cls.TASK_CONFIGS[top_task]["correction"].format(requested_module=requested)
