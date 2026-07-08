import json
import math
import random

import pytest

from src.tasks.config import TaskEvolutionConfig
from src.tasks.families import SINGLE_FILE_BUG_FIX
from src.tasks.task_pool import TaskPool

TRACEBACK_TEXT = "Traceback (most recent call last):\nTypeError: bad operand"
F2P = ["tests/test_app.py::test_a", "tests/test_app.py::test_b"]


def _patch(paths):
    return "".join(
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        for path in paths
    )


def make_instance(instance_id, files=("src/app.py",), statement=TRACEBACK_TEXT, f2p=None):
    return {
        "instance_id": instance_id,
        "repo": "org/proj",
        "base_commit": "deadbeef",
        "patch": _patch(files),
        "problem_statement": statement,
        "FAIL_TO_PASS": json.dumps(F2P if f2p is None else list(f2p)),
        "PASS_TO_PASS": json.dumps(["tests/test_app.py::test_ok"]),
    }


def make_instances():
    """One instance per family plus extras (families per classify_family)."""
    return [
        make_instance("org__single-1"),
        make_instance("org__single-2"),
        make_instance("org__multi-1", files=("src/a.py", "src/b.py", "src/c.py")),
        make_instance("org__config-1", files=("setup.py",)),
        make_instance(
            "org__aligned-1",
            statement="combine should return the fill value instead of NaN; expected behavior",
        ),
    ]


def accept_all(instance):
    return True, "ok"


def build_pool(tmp_path, name="task_pool.json", instances=None, verifier=accept_all):
    return TaskPool.from_instances(
        instances if instances is not None else make_instances(),
        tmp_path / name,
        config=TaskEvolutionConfig(),
        verifier=verifier,
    )


# ------------------------------------------------------------------ weights


def test_fresh_pool_has_unit_weights(tmp_path):
    pool = build_pool(tmp_path)
    assert len(pool) == 5
    assert all(weight == pytest.approx(1.0) for weight in pool.weights().values())


def test_family_ema_and_boundary_weight_math(tmp_path):
    pool = build_pool(tmp_path)
    pool.record_outcome("org__single-1", resolved=True, utility=0.8)

    ema = 0.7 * 0.5 + 0.3 * 1.0  # alpha=0.3, prior 0.5
    expected = math.exp(-((ema - 0.5) ** 2) / (2 * 0.2**2))
    assert pool.family_weight(SINGLE_FILE_BUG_FIX) == pytest.approx(expected)
    # Both single-file instances share the family weight.
    weights = pool.weights()
    assert weights["org__single-1"] == pytest.approx(expected)
    assert weights["org__single-2"] == pytest.approx(expected)
    # Other families are untouched.
    assert weights["org__multi-1"] == pytest.approx(1.0)


def test_utility_ema_is_tracked_in_state(tmp_path):
    pool = build_pool(tmp_path)
    pool.record_outcome("org__single-1", resolved=True, utility=0.8)
    pool.record_outcome("org__single-1", resolved=True, utility=0.4)
    state = json.loads(pool.save().read_text(encoding="utf-8"))

    assert state["instances"]["org__single-1"]["utility_ema"] == pytest.approx(
        0.7 * 0.8 + 0.3 * 0.4
    )
    assert state["instances"]["org__single-1"]["attempts"] == 2
    assert state["instances"]["org__single-1"]["resolved_count"] == 2


def test_record_outcome_unknown_instance_raises(tmp_path):
    with pytest.raises(KeyError, match="unknown instance"):
        build_pool(tmp_path).record_outcome("nope", resolved=True, utility=1.0)


# ---------------------------------------------------- too-hard decay/variants


def test_three_failures_decay_weight_and_emit_verified_variant(tmp_path):
    pool = build_pool(tmp_path)
    assert pool.record_outcome("org__single-1", resolved=False, utility=0.1) is None
    assert pool.record_outcome("org__single-1", resolved=False, utility=0.1) is None
    variant_id = pool.record_outcome("org__single-1", resolved=False, utility=0.1)

    assert variant_id == "org__single-1::focus-1"
    assert variant_id in pool
    assert len(pool) == 6
    variant = pool.get(variant_id)
    assert json.loads(variant["FAIL_TO_PASS"]) == [F2P[0]]
    assert pool.family_of(variant_id) == SINGLE_FILE_BUG_FIX
    # Same family, so the decay multiplier is exactly the weight ratio.
    weights = pool.weights()
    assert weights["org__single-1"] == pytest.approx(0.3 * weights[variant_id])


def test_variant_emitted_once_but_decay_compounds(tmp_path):
    pool = build_pool(tmp_path)
    for _ in range(3):
        pool.record_outcome("org__single-1", resolved=False, utility=0.0)
    for _ in range(2):
        assert pool.record_outcome("org__single-1", resolved=False, utility=0.0) is None
    assert pool.record_outcome("org__single-1", resolved=False, utility=0.0) is None

    assert len(pool) == 6  # focus_max_per_instance=1
    weights = pool.weights()
    ratio = weights["org__single-1"] / weights["org__single-1::focus-1"]
    assert ratio == pytest.approx(0.3**2)


def test_success_resets_consecutive_failures(tmp_path):
    pool = build_pool(tmp_path)
    for resolved in (False, False, True, False, False):
        pool.record_outcome("org__single-1", resolved=resolved, utility=0.2)

    assert len(pool) == 5  # never reached 3 consecutive failures
    weights = pool.weights()
    # No decay: both single-file instances still share the family weight.
    assert weights["org__single-1"] == pytest.approx(weights["org__single-2"])


def test_rejected_variant_is_not_admitted(tmp_path):
    pool = build_pool(tmp_path, verifier=lambda instance: (False, "image unresolvable"))
    for _ in range(3):
        pool.record_outcome("org__single-1", resolved=False, utility=0.0)
    assert len(pool) == 5


def test_no_verifier_means_no_variant(tmp_path):
    pool = build_pool(tmp_path, verifier=None)
    for _ in range(3):
        pool.record_outcome("org__single-1", resolved=False, utility=0.0)
    assert len(pool) == 5


def test_variants_do_not_spawn_variants(tmp_path):
    pool = build_pool(tmp_path)
    for _ in range(3):
        pool.record_outcome("org__single-1", resolved=False, utility=0.0)
    for _ in range(3):
        pool.record_outcome("org__single-1::focus-1", resolved=False, utility=0.0)
    assert len(pool) == 6


# ------------------------------------------------------------------ boosts


def test_apply_reflection_boosts(tmp_path):
    pool = build_pool(tmp_path)
    pool.apply_reflection(
        {
            "instance_boosts": {"org__single-1": 2, "unknown-id": 5},
            "family_boosts": {SINGLE_FILE_BUG_FIX: 1.5, "unknown_family": 9.0},
        }
    )
    weights = pool.weights()

    # instance: x(1 + 2*2), family: x1.5
    assert weights["org__single-1"] == pytest.approx(1.5 * 5.0)
    assert weights["org__single-2"] == pytest.approx(1.5)
    assert weights["org__multi-1"] == pytest.approx(1.0)

    # Signals are set, not accumulated: reapplying is idempotent.
    pool.apply_reflection({"instance_boosts": {"org__single-1": 2}})
    assert pool.weights()["org__single-1"] == pytest.approx(1.5 * 5.0)


# ---------------------------------------------------------------- sampling


def test_sampling_is_deterministic_for_a_fixed_seed(tmp_path):
    ids_by_run = []
    for run in range(2):
        pool = build_pool(tmp_path, name=f"pool_{run}.json")
        pool.record_outcome("org__single-1", resolved=True, utility=0.9)
        pool.record_outcome("org__multi-1", resolved=False, utility=0.1)
        sampled = pool.sample(4, random.Random(42))
        ids_by_run.append([instance["instance_id"] for instance in sampled])

    assert ids_by_run[0] == ids_by_run[1]
    assert len(set(ids_by_run[0])) == 4  # without replacement


def test_sample_edge_cases(tmp_path):
    pool = build_pool(tmp_path)
    assert pool.sample(0, random.Random(0)) == []
    everything = pool.sample(100, random.Random(0))
    assert len(everything) == len(pool)
    assert len({instance["instance_id"] for instance in everything}) == len(pool)


def test_zero_weight_instances_are_never_sampled(tmp_path):
    instances = [make_instance("org__single-1"), make_instance("org__config-1", files=("setup.py",))]
    pool = build_pool(tmp_path, instances=instances)
    pool.apply_reflection({"family_boosts": {SINGLE_FILE_BUG_FIX: 0.0}})

    for seed in range(10):
        sampled = pool.sample(1, random.Random(seed))
        assert sampled[0]["instance_id"] == "org__config-1"


# -------------------------------------------------------------- persistence


def _drive(pool):
    pool.record_outcome("org__single-1", resolved=True, utility=0.9)
    for _ in range(3):
        pool.record_outcome("org__multi-1", resolved=False, utility=0.1)
    pool.apply_reflection({"instance_boosts": {"org__config-1": 1}})
    return pool.save()


def test_state_json_is_reproducible_across_identical_runs(tmp_path):
    first = _drive(build_pool(tmp_path, name="a/task_pool.json")).read_bytes()
    second = _drive(build_pool(tmp_path, name="b/task_pool.json")).read_bytes()
    assert first == second


def test_state_roundtrip_restores_pool(tmp_path):
    pool = build_pool(tmp_path)
    _drive(pool)

    resumed = build_pool(tmp_path)  # same state_path -> resume
    assert len(resumed) == len(pool)
    assert "org__multi-1::focus-1" in resumed
    assert resumed.weights() == pytest.approx(pool.weights())

    # Focused variants only emit once, even across resume.
    for _ in range(3):
        resumed.record_outcome("org__multi-1", resolved=False, utility=0.0)
    assert len(resumed) == len(pool)


def test_stale_state_entries_are_dropped(tmp_path):
    pool = build_pool(tmp_path)
    _drive(pool)

    smaller = build_pool(tmp_path, instances=[make_instance("org__single-1")])
    # The persisted variant survives (it is standalone), stale ids are dropped.
    assert "org__multi-1::focus-1" in smaller
    assert "org__multi-1" not in smaller
    assert "org__single-1" in smaller
