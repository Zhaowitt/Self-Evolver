#!/usr/bin/env bash
# Run the stage scripts in order for one arm (EXPERIMENT + SEED). Every stage is
# resumable and can also be run on its own: bash scripts/<n>_<stage>.sh.
# STAGES default: check download splits sft grpo eval metrics (add "flaky" to use stage 3).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STAGES="${STAGES:-check download splits sft grpo eval metrics}"
for script in 0_check 1_download 2_splits 3_flaky 4_sft 5_grpo 6_eval 7_metrics; do
    case " ${STAGES} " in
        *" ${script#*_} "*)
            info "=== ${script} (${ARM_NAME}) ==="
            bash "${SCRIPTS_DIR}/${script}.sh"
            ;;
    esac
done
info "Pipeline finished for ${ARM_NAME} (${STAGES})"
