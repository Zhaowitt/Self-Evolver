#!/usr/bin/env bash
# Install Self-Evolver into the current Python environment and report the
# machine-specific paths and container engine the experiments will use.
#
# Knobs: PYTHON, EXTRAS (e.g. EXTRAS=dev), TEST_BACKEND.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXTRAS="${EXTRAS:-}"

require_python
info "Python: $(${PYTHON} --version 2>&1)"

if [ -n "${EXTRAS}" ]; then
    info "Installing self-evolver (editable) with extras: ${EXTRAS}"
    "${PYTHON}" -m pip install -e ".[${EXTRAS}]"
else
    info "Installing self-evolver (editable)"
    "${PYTHON}" -m pip install -e .
fi

# Seed a local .env from the template so the required keys are visible.
if [ ! -f "${REPO_ROOT}/.env" ]; then
    cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
    info "Created ${REPO_ROOT}/.env from .env.example — set OPENAI_API_KEY before generation runs."
fi

# Report the container engine situation without failing (host mode is allowed).
if command -v apptainer >/dev/null 2>&1; then
    info "Container engine: apptainer ($(apptainer --version 2>&1 | head -n1))"
elif command -v docker >/dev/null 2>&1; then
    info "Container engine: docker ($(docker --version 2>&1 | head -n1))"
else
    warn "No container engine found. Official-semantics grading needs apptainer or docker (or --test-backend host)."
fi

info "SWEBENCH_DATA_DIR = ${SWEBENCH_DATA_DIR}"
info "SIF_CACHE_DIR     = ${SIF_CACHE_DIR}"
info "Setup complete. Next: scripts/download_benchmarks.sh"
