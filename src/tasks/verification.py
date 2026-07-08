"""
Task verification (Proposal 2.7.1).

Only verified tasks enter the sampling distribution: the instance must have
a non-empty FAIL_TO_PASS set and its official container image reference must
be derivable by the configured container backend
(src/environment/test_backend.py).
"""

from __future__ import annotations

from typing import Any, Tuple

from src.tasks.variants import base_instance_id, fail_to_pass

_REQUIRED_FIELDS = ("instance_id", "repo", "base_commit")


def verify_task(instance: dict, backend: Any) -> Tuple[bool, str]:
    """
    Verify one instance against a container test backend.

    The backend must expose ``image_key(instance: dict) -> str`` returning the
    instance's official container image reference (or raising when no image
    spec is derivable) -- this is the ContainerTestBackend interface from
    src.environment.test_backend. Focused variants are probed with their
    ``base_instance_id`` because they run inside the parent's image.

    Returns (ok, reason).
    """
    for field in _REQUIRED_FIELDS:
        if not instance.get(field):
            return False, f"missing required field: {field}"

    tests = fail_to_pass(instance)
    if not tests:
        return False, "FAIL_TO_PASS is empty"

    image_key = getattr(backend, "image_key", None)
    if not callable(image_key):
        raise TypeError(
            "verify_task requires a container test backend exposing "
            "image_key(instance); got "
            f"{type(backend).__name__}. Use resolve_backend('apptainer'|"
            "'docker', instance) from src.environment.test_backend."
        )

    base_id = base_instance_id(instance)
    probe = dict(instance)
    probe["instance_id"] = base_id
    probe.pop("base_instance_id", None)
    try:
        image = image_key(probe)
    except Exception as exc:
        return False, f"container image unresolvable for {base_id}: {exc}"
    if not image:
        return False, f"container image unresolvable for {base_id}: empty reference"
    return True, f"image={image} fail_to_pass={len(tests)}"
