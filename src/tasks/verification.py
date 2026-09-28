"""Only tasks with non-empty FAIL_TO_PASS and a derivable container image enter sampling."""

from __future__ import annotations

from typing import Any, Tuple

from src.tasks.variants import base_instance_id, fail_to_pass

_REQUIRED_FIELDS = ("instance_id", "repo", "base_commit")


def verify_task(instance: dict, backend: Any) -> Tuple[bool, str]:
    """Returns (ok, reason); focused variants are probed with their parent's image."""
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
