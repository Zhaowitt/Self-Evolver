from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

from scripts.preflight_qwen35 import _inspect_vllm_model


def _install_fake_vllm(monkeypatch, registry, engine_args=None) -> None:
    vllm = ModuleType("vllm")
    model_executor = ModuleType("vllm.model_executor")
    models = ModuleType("vllm.model_executor.models")
    models.ModelRegistry = registry
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
    if engine_args is not None:
        engine = ModuleType("vllm.engine")
        arg_utils = ModuleType("vllm.engine.arg_utils")
        arg_utils.EngineArgs = engine_args
        monkeypatch.setitem(sys.modules, "vllm.engine", engine)
        monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)


def test_vllm_new_api_inspects_architectures(monkeypatch):
    calls = []

    def inspect_model_cls(architectures):
        calls.append(architectures)

    registry = SimpleNamespace(inspect_model_cls=inspect_model_cls)
    _install_fake_vllm(monkeypatch, registry)

    _inspect_vllm_model("Qwen/Qwen3.5-4B", ["Qwen3_5ForConditionalGeneration"])

    assert calls == [["Qwen3_5ForConditionalGeneration"]]


def test_vllm_020_api_builds_and_passes_model_config(monkeypatch):
    calls = []
    config = object()

    def inspect_model_cls(architectures, model_config):
        calls.append((architectures, model_config))

    class FakeEngineArgs:
        def __init__(self, *, model):
            assert model == "Qwen/Qwen3.5-4B"

        def create_model_config(self):
            return config

    registry = SimpleNamespace(inspect_model_cls=inspect_model_cls)
    _install_fake_vllm(monkeypatch, registry, FakeEngineArgs)

    _inspect_vllm_model("Qwen/Qwen3.5-4B", ["Qwen3_5ForConditionalGeneration"])

    assert calls == [(["Qwen3_5ForConditionalGeneration"], config)]
