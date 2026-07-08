# Inspect Before Editing

## Description

Repo-level repair skill for inspecting the task, failure evidence, and relevant code context before making any code edit.

## Purpose

Prevent premature or guess-based patches by ensuring the worker first understands the observed failure, the expected behavior, and the initially relevant files, tests, or symbols.

## When to Use

  * Use at the beginning of every repository repair task.
  * Use before modifying code, tests, configuration, or generated artifacts.
  * Use when a task includes an issue statement, traceback, failing test, assertion diff, reproduction command, or task wrapper constraint.

## How to Apply

  * Read the task statement and identify the requested behavior.
  * Inspect available failure evidence, such as the issue text, traceback, assertion diff, failing test name, or reproduction command.
  * Inspect the most relevant source files, tests, symbols, or call sites before editing.
  * Reproduce the failure when feasible within the task budget, or inspect the focused test that would reproduce it.
  * Before editing, state the observed failure and the expected behavior in concrete terms.

## Constraints

  * Do not edit code before inspecting relevant context.
  * Do not choose files to modify based only on filename guesses.
  * Do not ignore available failure evidence.
  * Avoid broad refactoring before the failure is understood.

Action type: INSPECT only.
