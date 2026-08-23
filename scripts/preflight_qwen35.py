#!/usr/bin/env python3
"""Fail fast when the external training stack cannot load Qwen3.5.

The check intentionally stops before downloading model weights or allocating
an EasyR1 worker.  It validates the Hugging Face configuration, Transformers
model class, vLLM registry, and the presence of a multimodal loader in the
external EasyR1/verl checkout installed in the active Python environment.
"""

from __future__ import annotations

import argparse
from importlib import metadata, util
from pathlib import Path
import sys
from typing import Iterable

import yaml


QWEN35_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
MULTIMODAL_LOADER_MARKERS = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForVision2Seq",
)


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _model_from_config(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        value = data["worker"]["actor"]["model"]["model_path"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} does not define worker.actor.model.model_path") from exc
    model = str(value).strip()
    if not model:
        raise ValueError(f"{path} contains an empty worker.actor.model.model_path")
    return model


def _as_architectures(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return []


def _check_easyr1_multimodal_loader() -> str | None:
    try:
        spec = util.find_spec("verl.workers.fsdp_workers")
    except (ImportError, ModuleNotFoundError) as exc:
        return f"EasyR1/verl is not importable: {exc}"
    if spec is None or not spec.origin:
        return "EasyR1/verl is not importable (verl.workers.fsdp_workers not found)"
    source_path = Path(spec.origin)
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"cannot inspect EasyR1 model loader at {source_path}: {exc}"
    if not any(marker in source for marker in MULTIMODAL_LOADER_MARKERS):
        return (
            "the installed EasyR1 worker has no recognized multimodal AutoModel "
            f"loader ({source_path})"
        )
    return None


def check(model: str) -> None:
    try:
        import transformers
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(f"Transformers is not importable: {exc}") from exc

    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(f"Transformers cannot load the config for {model!r}: {exc}") from exc

    architectures = _as_architectures(getattr(config, "architectures", None))
    is_qwen35 = bool(QWEN35_ARCHITECTURES.intersection(architectures)) or (
        "qwen3_5" in str(getattr(config, "model_type", "")).lower()
    )
    print(f"Controller model: {model}")
    print(f"Architectures: {architectures or ['<unspecified>']}")
    print(
        "Versions: "
        f"transformers={_package_version('transformers')}, "
        f"vllm={_package_version('vllm')}, "
        f"torch={_package_version('torch')}"
    )
    if not is_qwen35:
        print("Non-Qwen3.5 model selected; Qwen3.5-specific checks skipped.")
        return

    errors: list[str] = []
    missing_transformers_classes = [
        architecture
        for architecture in architectures
        if architecture in QWEN35_ARCHITECTURES and not hasattr(transformers, architecture)
    ]
    if missing_transformers_classes:
        errors.append("Transformers does not expose: " + ", ".join(missing_transformers_classes))

    try:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.inspect_model_cls(architectures)
    except Exception as exc:
        errors.append(f"vLLM cannot inspect {architectures}: {exc}")

    easyr1_error = _check_easyr1_multimodal_loader()
    if easyr1_error:
        errors.append(easyr1_error)

    if errors:
        details = "\n  - ".join(errors)
        raise RuntimeError(
            "Qwen3.5 preflight failed:\n"
            f"  - {details}\n"
            "Install a Qwen3.5-capable Transformers/vLLM/EasyR1 stack before "
            "requesting a full 64-step GPU job."
        )

    print("Qwen3.5 compatibility preflight passed (weights were not loaded).")
    print("Next run a one-step GPU smoke test before the 64-step round.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="EasyR1 YAML config")
    parser.add_argument("--model", help="Optional model ID/path overriding the YAML value")
    args = parser.parse_args()

    try:
        model = (args.model or _model_from_config(args.config)).strip()
        check(model)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
