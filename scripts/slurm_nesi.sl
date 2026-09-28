#!/bin/bash -e
# NeSI SLURM job running scripts/pipeline.sh. Submit from the repo root:
#   sbatch --account=<project> scripts/slurm_nesi.sl
# Knobs: REPO_ROOT, PROJECT_CODE, PERSIST_ROOT, SCRATCH_ROOT, CONDA_ENV,
#        EASYR1_DIR, STATE_ROOT, plus every pipeline.sh knob.

#SBATCH --job-name=se-qwen35-4b
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# sbatch runs a spooled copy, so BASH_SOURCE is not inside the repo.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT="${SLURM_SUBMIT_DIR}"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
[ -f "${REPO_ROOT}/scripts/pipeline.sh" ] || {
    echo "REPO_ROOT=${REPO_ROOT} is not the Self-Evolver checkout; submit from the repo root or export REPO_ROOT." >&2
    exit 1
}
PROJECT_CODE="${PROJECT_CODE:-${SLURM_JOB_ACCOUNT:-}}"
[ -n "${PROJECT_CODE}" ] || { echo "Submit with sbatch --account=<project> (or export PROJECT_CODE)." >&2; exit 1; }
PERSIST_ROOT="${PERSIST_ROOT:-/nesi/project/${PROJECT_CODE}/${USER}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/nesi/nobackup/${PROJECT_CODE}/${USER}}"
CONDA_ENV="${CONDA_ENV:-${PERSIST_ROOT}/envs/self-evolver}"
EASYR1_DIR="${EASYR1_DIR:-${PERSIST_ROOT}/EasyR1}"
STATE_ROOT="${STATE_ROOT:-${PERSIST_ROOT}/self_evolver_state}"

module purge
module load Miniforge3/25.3.1-0
# Pin CUDA to vLLM's compiled cu126 ABI, not NeSI's moving default.
module load CUDA/12.6.3
module load Apptainer
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "${CONDA_ENV}"; set -u   # some activate hooks read unset vars

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export SWEBENCH_DATA_DIR="${SWEBENCH_DATA_DIR:-${SCRATCH_ROOT}/benchmarks}"
export SIF_CACHE_DIR="${SIF_CACHE_DIR:-${SCRATCH_ROOT}/sif}"
export SELF_EVOLVER_EVAL_CACHE_DIR="${SELF_EVOLVER_EVAL_CACHE_DIR:-${SCRATCH_ROOT}/eval_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID:-local}/apptainer}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID:-local}/python}"

# All experiment state (splits, per-arm runs, checkpoints, skill banks).
export DATA_DIR="${DATA_DIR:-${STATE_ROOT}/data}"
export RUNS_DIR="${RUNS_DIR:-${STATE_ROOT}/runs}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${SCRATCH_ROOT}/self_evolver/workspace}"

export TEST_BACKEND="${TEST_BACKEND:-apptainer}"
export EASYR1_DIR
export EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
export CONTROLLER_MODEL_PATH="${CONTROLLER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
export EASYR1_GPU_MEMORY_UTILIZATION="${EASYR1_GPU_MEMORY_UTILIZATION:-0.2}"
export EASYR1_NUM_GPUS="${EASYR1_NUM_GPUS:-1}"
export EASYR1_TENSOR_PARALLEL_SIZE="${EASYR1_TENSOR_PARALLEL_SIZE:-1}"
export REWARD_WORKERS="${REWARD_WORKERS:-8}"   # 16 CPUs; 12 needs ~32 CPUs

# Rounds are resumable: resubmit with the same settings after a wall-time kill.
export STAGES="${STAGES:-check download splits sft grpo eval metrics}"
export EXPERIMENT="${EXPERIMENT:-full_method}"
export ROUNDS="${ROUNDS:-3}"
export STEPS_PER_ROUND="${STEPS_PER_ROUND:-64}"

mkdir -p \
    "${HF_HOME}" \
    "${SWEBENCH_DATA_DIR}" \
    "${SIF_CACHE_DIR}" \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}" \
    "${TMPDIR}" \
    "${DATA_DIR}" \
    "${RUNS_DIR}" \
    "${WORKSPACE_ROOT}"

cd "${REPO_ROOT}"
echo "Node: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "EasyR1: ${EASYR1_DIR}"
echo "Controller model: ${CONTROLLER_MODEL_PATH}"
echo "Experiment: ${EXPERIMENT} seed ${SEED:-0}; stages: ${STAGES}; ${ROUNDS} x ${STEPS_PER_ROUND} steps"
nvidia-smi

exec bash "${REPO_ROOT}/scripts/pipeline.sh"
