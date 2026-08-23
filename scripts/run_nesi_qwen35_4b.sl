#!/bin/bash -e
# NeSI/Mahuika: one 64-step Controller GRPO round with Qwen3.5-4B.
#
# Submit from any directory with:
#   sbatch /path/to/Self-Evolver/scripts/run_nesi_qwen35_4b.sl
#
# The active Conda environment must already contain a Qwen3.5-capable EasyR1,
# Transformers, vLLM, PyTorch, and this repository.  The script intentionally
# runs the compatibility preflight before EasyR1 allocates the model.
#
# For the isolated one-step smoke command, see docs/NESI_QWEN35_RUNBOOK.md.

#SBATCH --account=uoa04658
#SBATCH --job-name=se-qwen35-4b
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_CODE="${PROJECT_CODE:-${SLURM_JOB_ACCOUNT:-uoa04658}}"
PERSIST_ROOT="${PERSIST_ROOT:-/nesi/project/${PROJECT_CODE}/${USER}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/nesi/nobackup/${PROJECT_CODE}/${USER}}"
CONDA_ENV="${CONDA_ENV:-${PERSIST_ROOT}/envs/self-evolver}"
EASYR1_DIR="${EASYR1_DIR:-${PERSIST_ROOT}/EasyR1}"
STATE_ROOT="${STATE_ROOT:-${PERSIST_ROOT}/self_evolver_state}"
ROUND_ID="${ROUND_ID:-qwen35-round01}"

module purge
module load Miniforge3/25.3.1-0
module load CUDA
module load Apptainer
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export SWEBENCH_DATA_DIR="${SWEBENCH_DATA_DIR:-${SCRATCH_ROOT}/benchmarks}"
export SIF_CACHE_DIR="${SIF_CACHE_DIR:-${SCRATCH_ROOT}/sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRATCH_ROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}/apptainer}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}/python}"

export DATA_DIR="${DATA_DIR:-${STATE_ROOT}/data}"
export RUNS_DIR="${RUNS_DIR:-${STATE_ROOT}/runs/${ROUND_ID}}"
export HARD_CASES_PATH="${HARD_CASES_PATH:-${STATE_ROOT}/hard_cases.jsonl}"
export SELF_EVOLVER_CLUSTER_REGISTRY="${SELF_EVOLVER_CLUSTER_REGISTRY:-${STATE_ROOT}/cluster_registry.json}"
export SELF_EVOLVER_CLUSTER_BASELINES="${SELF_EVOLVER_CLUSTER_BASELINES:-${STATE_ROOT}/cluster_baselines.json}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${SCRATCH_ROOT}/self_evolver/workspace/${ROUND_ID}}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-${WORKSPACE_ROOT}}"
export SELF_EVOLVER_REWARD_WORKSPACE="${SELF_EVOLVER_REWARD_WORKSPACE:-${SCRATCH_ROOT}/self_evolver/online_reward}"

export TEST_BACKEND="${TEST_BACKEND:-apptainer}"
export EASYR1_DIR
export EASYR1_CONFIG="${EASYR1_CONFIG:-${REPO_ROOT}/configs/easyr1_online_grpo_example.yaml}"
export CONTROLLER_MODEL_PATH="${CONTROLLER_MODEL_PATH:-Qwen/Qwen3.5-4B}"
export QWEN35_PREFLIGHT="${QWEN35_PREFLIGHT:-1}"
export EASYR1_GPU_MEMORY_UTILIZATION="${EASYR1_GPU_MEMORY_UTILIZATION:-0.2}"
export EASYR1_NUM_GPUS="${EASYR1_NUM_GPUS:-1}"
export EASYR1_TENSOR_PARALLEL_SIZE="${EASYR1_TENSOR_PARALLEL_SIZE:-1}"

export NUM_TRAIN="${NUM_TRAIN:-320}"
export NUM_VAL="${NUM_VAL:-18}"
export REWARD_WORKERS="${REWARD_WORKERS:-4}"
export RUN_EVAL="${RUN_EVAL:-0}"
export REBUILD_DATASETS="${REBUILD_DATASETS:-1}"
export ROUND_ID
export EASYR1_EXPERIMENT_NAME="${EASYR1_EXPERIMENT_NAME:-controller_qwen3_5_4b_${ROUND_ID}}"
# max_steps is cumulative. When loading global_step_960, set this to 1024.
export EASYR1_MAX_STEPS="${EASYR1_MAX_STEPS:-64}"

mkdir -p \
    "${HF_HOME}" \
    "${SWEBENCH_DATA_DIR}" \
    "${SIF_CACHE_DIR}" \
    "${APPTAINER_CACHEDIR}" \
    "${APPTAINER_TMPDIR}" \
    "${TMPDIR}" \
    "${DATA_DIR}" \
    "${RUNS_DIR}" \
    "${WORKSPACE_ROOT}" \
    "${SELF_EVOLVER_REWARD_WORKSPACE}"

cd "${REPO_ROOT}"
echo "Node: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "EasyR1: ${EASYR1_DIR}"
echo "Controller model: ${CONTROLLER_MODEL_PATH}"
echo "Round: ${ROUND_ID}; cumulative max steps: ${EASYR1_MAX_STEPS}"
nvidia-smi

exec bash "${REPO_ROOT}/scripts/run_rl_controller.sh"
