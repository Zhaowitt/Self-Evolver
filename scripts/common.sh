#!/usr/bin/env bash
# Shared helpers for the Self-Evolver experiment scripts.
#
# Every experiment script sources this file. It derives the repository root from
# its own location (no absolute paths baked in), loads an optional .env, sets
# portable defaults for the machine-specific knobs, and exposes small pre-flight
# checks so each script can fail early with an actionable message.
#
# Override any knob from the environment, e.g.
#   TEST_BACKEND=docker SEED=1 RUNS_DIR=/data/runs scripts/run_full_method.sh

# Resolve the repo root from this file's directory (scripts/ -> repo root).
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"
export REPO_ROOT

# Run everything from the repo root so `python -m src...` resolves.
cd "${REPO_ROOT}"

# --- logging helpers ----------------------------------------------------------
info()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn()  { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()   { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# --- optional .env ------------------------------------------------------------
# Load KEY=VALUE lines from .env without clobbering variables already exported
# in the shell (the shell wins, matching dotenv semantics).
load_dotenv() {
    local env_file="${REPO_ROOT}/.env"
    [ -f "${env_file}" ] || return 0
    local line key val
    while IFS= read -r line || [ -n "${line}" ]; do
        line="${line%%#*}"                       # strip trailing comments
        line="${line#"${line%%[![:space:]]*}"}"  # ltrim
        line="${line%"${line##*[![:space:]]}"}"  # rtrim
        [ -z "${line}" ] && continue
        case "${line}" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"     # rtrim key
        val="${val#\"}"; val="${val%\"}"          # strip one layer of quotes
        val="${val#\'}"; val="${val%\'}"
        case "${val}" in "~/"*) val="${HOME}/${val#\~/}" ;; esac
        if [ -z "${!key:-}" ]; then
            export "${key}=${val}"
        fi
    done < "${env_file}"
}
load_dotenv

# --- portable defaults --------------------------------------------------------
PYTHON="${PYTHON:-python}"
BENCHMARK="${BENCHMARK:-swebench}"          # swebench | swebench_live | swebench_pro | multi_swe_bench
TEST_BACKEND="${TEST_BACKEND:-apptainer}"   # auto | docker | apptainer | host (HPC default: apptainer)
SEED="${SEED:-0}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_NAME="${MODEL_NAME:-self-evolver}"

# Data + SIF locations are read by the Python code; export them so a child
# process picks up the same paths (Python expands a leading ~).
export SWEBENCH_DATA_DIR="${SWEBENCH_DATA_DIR:-${REPO_ROOT}/benchmarks}"
export SIF_CACHE_DIR="${SIF_CACHE_DIR:-${HOME}/.cache/self_evolver/sif}"

# --- pre-flight checks --------------------------------------------------------
require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found on PATH. $2"
}

require_python() {
    require_cmd "${PYTHON}" "Set PYTHON=/path/to/python."
    "${PYTHON}" - <<'PY' || die "Python >= 3.10 is required."
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)
PY
}

require_container_engine() {
    case "${TEST_BACKEND}" in
        host) return 0 ;;
        docker)    require_cmd docker    "Install Docker or set TEST_BACKEND=apptainer." ;;
        apptainer) require_cmd apptainer "Install Apptainer or set TEST_BACKEND=docker." ;;
        auto)
            command -v docker >/dev/null 2>&1 || command -v apptainer >/dev/null 2>&1 || \
                die "No container engine found. Install docker or apptainer, or set TEST_BACKEND=host."
            ;;
        *) die "Unknown TEST_BACKEND='${TEST_BACKEND}' (auto|docker|apptainer|host)." ;;
    esac
}

require_api_key() {
    [ -n "${OPENAI_API_KEY:-}" ] || die \
        "OPENAI_API_KEY is not set. Export it or add it to ${REPO_ROOT}/.env (see .env.example). Patch generation calls the worker LLM."
}

require_data() {
    [ -d "${SWEBENCH_DATA_DIR}" ] || die \
        "Benchmark data dir '${SWEBENCH_DATA_DIR}' is missing. Run scripts/download_benchmarks.sh (or set SWEBENCH_DATA_DIR)."
}

# Reset the on-disk skill bank to the seed state so an evolution experiment
# starts clean (opt-in: scripts call this only when RESET_SKILLS=1).
reset_skill_bank() {
    local skills_dir="${REPO_ROOT}/skills"
    info "Resetting skill bank runtime state under ${skills_dir}"
    rm -f "${skills_dir}/metadata.json"
    rm -rf "${skills_dir}/_archive"
    if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        git -C "${REPO_ROOT}" checkout -- skills 2>/dev/null || true
        git -C "${REPO_ROOT}" clean -fdq skills 2>/dev/null || true
    fi
}

# Run one benchmark configuration. First arg is the run directory; the rest are
# passed straight through to `python -m src.main benchmark`.
run_benchmark() {
    local run_dir="$1"; shift
    mkdir -p "${run_dir}" "${WORKSPACE_ROOT}"
    info "benchmark -> ${run_dir}"
    "${PYTHON}" -m src.main benchmark \
        --benchmark "${BENCHMARK}" \
        --output-dir "${run_dir}" \
        --workspace-dir "${WORKSPACE_ROOT}" \
        --test-backend "${TEST_BACKEND}" \
        --seed "${SEED}" \
        --model-name "${MODEL_NAME}" \
        "$@"
}
