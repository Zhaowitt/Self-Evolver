# Existing Pattern Alignment

## Description

Repo-level repair skill for making a fix follow the repository's existing implementation, test, style, naming, and error-handling patterns.

## Purpose

Keep patches compatible with the surrounding project by reusing established conventions instead of introducing unnecessary abstractions, dependencies, or one-off designs.

## When to Use

  * Use after localizing the likely failure region and before writing a repair plan.
  * Use when nearby code, related tests, helper functions, or similar bug fixes can guide the patch.
  * Use when the localized fix could be implemented in multiple ways and the worker must choose the repo-native one.

## How to Apply

  * Inspect nearby implementations that solve similar problems.
  * Inspect related tests to learn the expected behavior and assertion style.
  * Reuse existing helpers, data structures, naming conventions, control flow, and error-handling patterns when appropriate.
  * Match the surrounding file's level of abstraction and avoid stylistic cleanup unrelated to the repair.
  * Introduce a new helper or abstraction only when existing patterns are insufficient for the localized fix.

## Constraints

  * Do not introduce a new pattern when an existing repository pattern is available.
  * Do not change public behavior outside the localized failure path without justification.
  * Do not add dependencies, broad abstractions, or large rewrites for a localized issue.
  * Do not perform unrelated formatting or cleanup.

Action type: ALIGN only.
