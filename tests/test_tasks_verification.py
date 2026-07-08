import json

import pytest

from src.tasks.variants import make_focused_variant
from src.tasks.verification import verify_task


class FakeBackend:
    """
    Scripted container backend double (tests only).

    Mirrors src.environment.test_backend.ContainerTestBackend.image_key:
    returns the official image reference, or raises when no spec is derivable.
    """

    def __init__(self, fail_ids=(), empty_ids=()):
        self.fail_ids = set(fail_ids)
        self.empty_ids = set(empty_ids)
        self.calls = []

    def image_key(self, instance):
        instance_id = instance["instance_id"]
        self.calls.append(instance_id)
        if instance_id in self.fail_ids:
            raise KeyError("version")
        if instance_id in self.empty_ids:
            return ""
        return f"swebench/sweb.eval.x86_64.{instance_id}:latest"


def make_instance(instance_id="org__proj-1", f2p=("tests/test_app.py::test_a",)):
    return {
        "instance_id": instance_id,
        "repo": "org/proj",
        "base_commit": "deadbeef",
        "problem_statement": "The app crashes.",
        "FAIL_TO_PASS": json.dumps(list(f2p)),
    }


def test_verify_task_accepts_resolvable_instance():
    backend = FakeBackend()
    ok, reason = verify_task(make_instance(), backend)

    assert ok
    assert "sweb.eval.x86_64.org__proj-1" in reason
    assert backend.calls == ["org__proj-1"]


def test_verify_task_rejects_empty_fail_to_pass():
    ok, reason = verify_task(make_instance(f2p=()), FakeBackend())
    assert not ok
    assert "FAIL_TO_PASS" in reason


def test_verify_task_rejects_missing_required_fields():
    instance = make_instance()
    del instance["repo"]
    ok, reason = verify_task(instance, FakeBackend())
    assert not ok
    assert "repo" in reason


def test_verify_task_rejects_unresolvable_image():
    ok, reason = verify_task(make_instance(), FakeBackend(fail_ids={"org__proj-1"}))
    assert not ok
    assert "unresolvable" in reason
    assert "version" in reason


def test_verify_task_rejects_empty_image_reference():
    ok, reason = verify_task(make_instance(), FakeBackend(empty_ids={"org__proj-1"}))
    assert not ok
    assert "unresolvable" in reason


def test_verify_task_probes_focused_variant_with_base_image():
    parent = make_instance(f2p=("tests/test_a", "tests/test_b"))
    variant = make_focused_variant(parent)
    backend = FakeBackend()

    ok, _ = verify_task(variant, backend)

    assert ok
    assert backend.calls == ["org__proj-1"]


def test_verify_task_requires_image_key_interface():
    class NotABackend:
        pass

    with pytest.raises(TypeError, match="image_key"):
        verify_task(make_instance(), NotABackend())


def test_container_backend_exposes_the_image_key_interface():
    """verify_task depends on ContainerTestBackend.image_key(instance) -> str."""
    from src.environment.test_backend import ContainerTestBackend

    assert callable(getattr(ContainerTestBackend, "image_key", None))
