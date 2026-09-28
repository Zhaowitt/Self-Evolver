#!/usr/bin/env bash
# Shared helpers sourced by every script: repo root, .env, defaults, checks.
# Knobs: PYTHON, BENCHMARK, TEST_BACKEND, SEED, RUNS_DIR, WORKSPACE_ROOT, DATA_DIR,
#        MODEL_NAME, SWEBENCH_DATA_DIR, SIF_CACHE_DIR, FRESH_RUN.

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
export REPO_ROOT

# `python -m src...` must resolve from the repo root.
cd "${REPO_ROOT}"

info()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn()  { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()   { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Load KEY=VALUE lines from .env; already-exported shell variables win.
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

# EasyR1 runs the reward function from its own checkout, so repo-relative paths
# must be absolute for every process to share the same state.
absolutize_path_vars() {
    local name val
    for name in WORKSPACE_DIR SWEBENCH_DATA_DIR SIF_CACHE_DIR SELF_EVOLVER_EVAL_CACHE_DIR \
        SELF_EVOLVER_HARD_CASES_PATH SELF_EVOLVER_CLUSTER_REGISTRY SELF_EVOLVER_CLUSTER_BASELINES \
        SELF_EVOLVER_REWARD_WORKSPACE SELF_EVOLVER_ROLLOUT_JSONL SELF_EVOLVER_SKILLS_DIR; do
        val="${!name:-}"
        [ -n "${val}" ] || continue
        case "${val}" in
            /*) ;;
            *) export "${name}=${REPO_ROOT}/${val#./}" ;;
        esac
    done
}
absolutize_path_vars

PYTHON="${PYTHON:-python}"
BENCHMARK="${BENCHMARK:-swebench}"          # swebench | swebench_live | swebench_pro | multi_swe_bench
TEST_BACKEND="${TEST_BACKEND:-apptainer}"   # auto | docker | apptainer | host (HPC default: apptainer)
SEED="${SEED:-0}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${REPO_ROOT}/workspace}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_NAME="${MODEL_NAME:-self-evolver}"

# Paths shared by the stage scripts (one experiment arm = EXPERIMENT + SEED).
export EXPERIMENT="${EXPERIMENT:-full_method}"
export ARM_NAME="${ARM_NAME:-${EXPERIMENT}-seed${SEED}}"
export ARM_DIR="${ARM_DIR:-${RUNS_DIR}/grpo/${ARM_NAME}}"
export SPLITS_DIR="${SPLITS_DIR:-${DATA_DIR}/splits}"
export SFT_DIR="${SFT_DIR:-${RUNS_DIR}/controller_sft}"
export FLAKY_IDS="${FLAKY_IDS:-${SPLITS_DIR}/flaky_ids.txt}"

# Read by the Python code (which expands a leading ~).
export SWEBENCH_DATA_DIR="${SWEBENCH_DATA_DIR:-${REPO_ROOT}/benchmarks}"
export SIF_CACHE_DIR="${SIF_CACHE_DIR:-${HOME}/.cache/self_evolver/sif}"

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
    [ "${OPENAI_API_KEY}" != "sk-your-api-key-here" ] || die \
        "OPENAI_API_KEY is still the .env.example placeholder; set a real key."
}

# Append --exclude-ids args for each id file in $2 (whitespace-separated, or one
# path with spaces) to the array named $1. Missing files are fatal unless
# $3=optional: a missing contamination list must not leak eval ids into training.
exclude_id_args() {
    local -n _out="$1"
    local list="$2" mode="${3:-required}" path
    local -a paths=()
    if [ -f "${list}" ]; then
        paths=("${list}")
    else
        read -ra paths <<< "${list}"
    fi
    for path in "${paths[@]}"; do
        if [ -f "${path}" ]; then
            _out+=(--exclude-ids "${path}")
            info "Excluding ids in ${path}"
        elif [ "${mode}" = "optional" ]; then
            warn "Optional id file ${path} not found; skipped"
        else
            die "Exclusion id file ${path} not found. Run scripts/2_splits.sh or set ALLOW_MISSING_EXCLUDE_IDS=1."
        fi
    done
}

require_data() {
    [ -d "${SWEBENCH_DATA_DIR}" ] || die \
        "Benchmark data dir '${SWEBENCH_DATA_DIR}' is missing. Run scripts/1_download.sh (or set SWEBENCH_DATA_DIR)."
}

# Tracked skills/ is the read-only seed bank; experiments evolve their own copies.
SEED_SKILLS_DIR="${REPO_ROOT}/skills"

# Point this experiment at its own bank (seeded once) so evolved skills never
# leak into another run's baseline.
use_skill_bank() {
    local skills_dir="$1"
    if [ ! -e "${skills_dir}/.initialized" ]; then
        mkdir -p "${skills_dir}"
        cp -a "${SEED_SKILLS_DIR}/." "${skills_dir}/"
        touch "${skills_dir}/.initialized"
        info "Initialized skill bank ${skills_dir} from ${SEED_SKILLS_DIR}"
    fi
    export SELF_EVOLVER_SKILLS_DIR="${skills_dir}"
}

# Restore SELF_EVOLVER_SKILLS_DIR to the seed skills (never the seed itself).
reset_skill_bank() {
    local skills_dir="${SELF_EVOLVER_SKILLS_DIR:-}"
    [ -n "${skills_dir}" ] || die "reset_skill_bank needs SELF_EVOLVER_SKILLS_DIR (call use_skill_bank first)."
    [ "$(cd "${skills_dir}" 2>/dev/null && pwd)" != "$(cd "${SEED_SKILLS_DIR}" && pwd)" ] || \
        die "Refusing to reset the seed bank itself (${SEED_SKILLS_DIR})."
    info "Resetting skill bank ${skills_dir} to the seed skills"
    rm -rf "${skills_dir}"
    mkdir -p "${skills_dir}"
    cp -a "${SEED_SKILLS_DIR}/." "${skills_dir}/"
    touch "${skills_dir}/.initialized"
}

# Ray, vLLM (zmq) and multiprocessing put AF_UNIX sockets (max 107 bytes)
# under TMPDIR / RAY_TMPDIR; long job TMPDIRs make them fail to start.
ensure_short_tmpdir() {
    export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER:-self-evolver}}"
    if [ -n "${TMPDIR:-}" ] && [ "${#TMPDIR}" -gt 60 ]; then
        warn "TMPDIR is too long for AF_UNIX sockets; using ${RAY_TMPDIR}/tmp"
        export TMPDIR="${RAY_TMPDIR}/tmp"
    fi
    mkdir -p "${RAY_TMPDIR}" ${TMPDIR:+"${TMPDIR}"}
}
ensure_short_tmpdir

# run_benchmark RUN_DIR [src.main benchmark args...]
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
        ${FRESH_RUN:+--no-resume} \
        "$@"
}
