"""Structured diagnosis derived from an entire failed repair trajectory."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List


FAILURE_MECHANISMS = {
    "localization_failure",
    "wrong_root_cause",
    "patch_generation_failure",
    "patch_application_failure",
    "incomplete_fix",
    "regression_introduced",
    "test_interpretation_failure",
    "environment_failure",
    "unknown",
}

_ENVIRONMENT_TERMS = (
    "modulenotfounderror",
    "no module named",
    "dependency",
    "docker",
    "apptainer",
    "container",
    "image pull",
    "network",
    "permission denied",
    "tests_timeout",
    "timed out",
)


@dataclass
class FailureDiagnosis:
    mechanism: str = "unknown"
    summary: str = ""
    error_signatures: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "rules_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def diagnose_failed_trajectory(
    records: Iterable[Any],
    termination_reason: str,
) -> FailureDiagnosis:
    """Create one primary mechanism plus compact, fielded evidence."""
    attempts = list(records)
    statuses: List[str] = []
    routes: List[str] = []
    patch_applied: List[bool] = []
    f2p_progress: List[Dict[str, Any]] = []
    p2p_regressions: List[Dict[str, Any]] = []
    raw_errors: List[str] = []

    for index, record in enumerate(attempts, start=1):
        if getattr(record, "error", None):
            raw_errors.append(str(record.error))
        decision = getattr(record, "judge_decision", None)
        if decision:
            route = getattr(getattr(decision, "route", None), "value", "")
            if route:
                routes.append(route)
            feedback = getattr(decision, "feedback_for_next_attempt", "")
            if feedback:
                raw_errors.append(str(feedback))
        verification = getattr(record, "verification_result", None)
        if not verification:
            continue
        status = getattr(getattr(verification, "status", None), "value", "")
        if status:
            statuses.append(status)
        patch_applied.append(bool(getattr(verification, "patch_applied", False)))
        for name in ("summary", "error_message"):
            value = getattr(verification, name, "")
            if value:
                raw_errors.append(str(value))

        outcomes = []
        subset = getattr(verification, "subset_eval_outcome", None)
        full = getattr(verification, "full_eval_outcome", None) or getattr(
            verification, "eval_outcome", None
        )
        if subset is not None:
            outcomes.append(("subset", subset))
        if full is not None:
            outcomes.append(("full", full))
        for scope, outcome in outcomes:
            passed = int(getattr(outcome, "f2p_passed", 0) or 0)
            total = int(getattr(outcome, "f2p_total", 0) or 0)
            f2p_progress.append(
                {
                    "iteration": index,
                    "scope": scope,
                    "passed": passed,
                    "total": total,
                    "fraction": round(passed / total, 4) if total else 0.0,
                }
            )
            p2p_passed = int(getattr(outcome, "p2p_passed", 0) or 0)
            p2p_total = int(getattr(outcome, "p2p_total", 0) or 0)
            if scope == "full" and p2p_passed < p2p_total:
                p2p_regressions.append(
                    {
                        "iteration": index,
                        "passed": p2p_passed,
                        "total": p2p_total,
                    }
                )

    signatures = extract_error_signatures(raw_errors)
    evidence = {
        "f2p_progress": f2p_progress,
        "p2p_regression": p2p_regressions,
        "patch_applied": patch_applied,
        "retry_pattern": {"statuses": statuses, "routes": routes},
        "termination_reason": termination_reason,
    }
    mechanism, confidence = _rule_mechanism(
        statuses=statuses,
        routes=routes,
        patch_applied=patch_applied,
        f2p_progress=f2p_progress,
        p2p_regressions=p2p_regressions,
        signatures=signatures,
    )
    return FailureDiagnosis(
        mechanism=mechanism,
        summary=_summary(mechanism, statuses, f2p_progress, p2p_regressions),
        error_signatures=signatures,
        evidence=evidence,
        confidence=confidence,
    )


def adjudicate_ambiguous_diagnoses(records: List[Any], llm_client: Any) -> List[Any]:
    """Batch-classify rule-ambiguous v2 records after the round has frozen."""
    ambiguous = [
        (index, record)
        for index, record in enumerate(records)
        if record.failure_diagnosis
        and (
            (record.failure_diagnosis or {}).get("mechanism") == "unknown"
            or float((record.failure_diagnosis or {}).get("confidence", 0.0) or 0.0) < 0.5
        )
    ]
    if not ambiguous or llm_client is None:
        return records
    payload = [
        {
            "record_index": index,
            "issue_id": record.issue_id,
            "repo_name": record.repo_name,
            "diagnosis": record.failure_diagnosis,
            "verification_statuses": record.verification_statuses,
            "errors": record.errors[:5],
        }
        for index, record in ambiguous
    ]
    system = (
        "Classify ambiguous failed coding trajectories. Return JSON only as "
        "{\"diagnoses\":[{\"record_index\":0,\"mechanism\":\"...\","
        "\"summary\":\"evidence-grounded pattern\",\"confidence\":0.0}]}. "
        f"mechanism must be one of {sorted(FAILURE_MECHANISMS - {'unknown'})}. "
        "max_iterations and tests_failed are outcomes, never mechanisms."
    )
    try:
        response = llm_client.chat_with_system(system, json.dumps(payload, ensure_ascii=False))
        text = str(getattr(response, "content", "") or "")
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start : end + 1])
    except Exception:
        return records
    by_index = {index: record for index, record in ambiguous}
    for item in data.get("diagnoses") or []:
        try:
            index = int(item.get("record_index"))
            mechanism = str(item.get("mechanism") or "")
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if index not in by_index or mechanism not in FAILURE_MECHANISMS - {"unknown"}:
            continue
        diagnosis = dict(by_index[index].failure_diagnosis)
        diagnosis.update(
            {
                "mechanism": mechanism,
                "summary": str(item.get("summary") or diagnosis.get("summary") or "")[:500],
                "confidence": max(0.0, min(1.0, confidence)),
                "source": "llm_adjudication_v1",
            }
        )
        by_index[index].failure_diagnosis = diagnosis
        by_index[index].failure_type = mechanism
    return records


def extract_error_signatures(values: Iterable[str], limit: int = 6) -> List[str]:
    """Normalize concrete error lines without retaining whole noisy logs."""
    signatures: List[str] = []
    for value in values:
        for line in str(value or "").splitlines():
            line = re.sub(r"[A-Za-z]:[/\\][^\s:]+", "<path>", line.strip())
            line = re.sub(r"/[^\s:]+", "<path>", line)
            line = re.sub(r"\bline \d+\b", "line <n>", line, flags=re.I)
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) < 8:
                continue
            signature = line[:240]
            if signature not in signatures:
                signatures.append(signature)
            if len(signatures) >= limit:
                return signatures
    return signatures


def _rule_mechanism(
    statuses: List[str],
    routes: List[str],
    patch_applied: List[bool],
    f2p_progress: List[Dict[str, Any]],
    p2p_regressions: List[Dict[str, Any]],
    signatures: List[str],
) -> tuple[str, float]:
    signature_text = " ".join(signatures).lower()
    if any(term in signature_text for term in _ENVIRONMENT_TERMS):
        return "environment_failure", 0.9
    if p2p_regressions or "new_issues" in statuses:
        return "regression_introduced", 0.95
    if statuses and all(status in {"patch_failed", "no_changes"} for status in statuses):
        return "patch_application_failure", 0.95
    if statuses and all(status == "empty_patch" for status in statuses):
        return "patch_generation_failure", 0.95
    if not statuses and any("patch generation failed" in item.lower() for item in signatures):
        return "patch_generation_failure", 0.85

    fractions = [float(item.get("fraction", 0.0)) for item in f2p_progress]
    if any(0.0 < fraction < 1.0 for fraction in fractions):
        return "incomplete_fix", 0.9
    if any(item.get("scope") == "subset" for item in f2p_progress) and any(
        item.get("scope") == "full" and item.get("fraction", 0.0) < 1.0
        for item in f2p_progress
    ):
        return "test_interpretation_failure", 0.75
    if patch_applied and any(patch_applied) and fractions and max(fractions) == 0.0:
        if routes.count("reinspect") >= 1:
            return "wrong_root_cause", 0.8
        return "incomplete_fix", 0.7
    if routes.count("reinspect") >= 2 or (not patch_applied and routes):
        return "localization_failure", 0.7
    if "patch_failed" in statuses or "no_changes" in statuses:
        return "patch_application_failure", 0.75
    if "empty_patch" in statuses:
        return "patch_generation_failure", 0.75
    return "unknown", 0.3


def _summary(
    mechanism: str,
    statuses: List[str],
    progress: List[Dict[str, Any]],
    regressions: List[Dict[str, Any]],
) -> str:
    status_pattern = " -> ".join(statuses) or "no verifier result"
    full = [item for item in progress if item.get("scope") == "full"]
    subset = [item for item in progress if item.get("scope") == "subset"]
    details: List[str] = [f"status pattern: {status_pattern}"]
    if subset:
        details.append(
            "subset F2P: " + ", ".join(f"{x['passed']}/{x['total']}" for x in subset)
        )
    if full:
        details.append("full F2P: " + ", ".join(f"{x['passed']}/{x['total']}" for x in full))
    if regressions:
        details.append("P2P regression observed")
    return f"{mechanism}: " + "; ".join(details)
