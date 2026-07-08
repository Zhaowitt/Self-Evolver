# Repair Plan Before Patch

## Description

Repo-level repair skill for forming a concrete causal repair plan before editing code.

## Purpose

Prevent trial-and-error patching by requiring the worker to connect the observed failure, localized root cause, existing repository pattern, intended edit, and validation signal.

## When to Use

  * Use immediately before making a code edit.
  * Use after inspecting the failure, localizing the likely root cause, and checking relevant repository patterns.
  * Use when the worker is ready to convert analysis into a patch.

## How to Apply

  * State the localized root cause or most likely causal mechanism.
  * State the existing repository pattern the patch should follow.
  * Identify the file, function, class, or symbol to edit.
  * Describe the intended change in one or two concrete steps.
  * Identify the most relevant validation signal, such as a focused test, expected assertion behavior, reproduction command, or static check.

## Constraints

  * Do not edit repeatedly without a stated repair hypothesis.
  * Do not make a patch that does not follow from the localized failure.
  * Do not plan a broad rewrite when a localized change is plausible.
  * Do not patch without knowing how the change should be validated.

Action type: PLAN only.
