# Failure Localization

## Description

Repo-level repair skill for narrowing an observed failure to the smallest plausible root-cause region before planning a patch.

## Purpose

Reduce broad, unrelated, or symptom-only changes by connecting the failure to a specific code path, function, class, module, data flow, or test behavior.

## When to Use

  * Use after initial inspection and before deciding what to patch.
  * Use when the failure can be traced through tests, stack traces, call sites, imports, or related symbols.
  * Use when several files or modules appear relevant and the worker must avoid unnecessary edits.

## How to Apply

  * Trace the failure from the observed symptom toward the likely implementation point.
  * Use focused search, stack traces, call sites, tests, and nearby implementations to narrow the candidate region.
  * Distinguish the root cause from downstream symptoms or error-reporting locations.
  * Prefer the smallest plausible fault region over module-wide assumptions.
  * State why the selected region is more likely causal than nearby alternatives.

## Constraints

  * Do not modify several unrelated areas without a localization hypothesis.
  * Do not treat the visible error location as the root cause without checking relevant callers or data flow.
  * Do not silence a symptom while leaving the likely root cause unaddressed.
  * Do not expand the patch scope before exhausting the most relevant localized path.

Action type: LOCALIZE only.
