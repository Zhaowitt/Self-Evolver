"""Regression guards for packaging, docs, and the experiment scripts.

These are pure file-parsing checks (plus ``bash -n``): no network, LLM,
containers, or GPU. They lock in the fixes for the stale ``.env.example``, the
dead dependencies, the ``*.json`` gitignore collision, and the script hygiene
rules (portable ``REPO_ROOT`` derivation, strict mode, no absolute paths).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Direct environment reads: os.getenv("X") / os.environ.get("X") / os.environ["X"].
_ENV_READ = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z0-9_]+)["']"""
    r"""|os\.environ\[\s*["']([A-Z0-9_]+)["']"""
)


def _env_example_keys() -> set[str]:
    keys = set()
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


def _directly_read_env_vars() -> set[str]:
    found: set[str] = set()
    for path in (REPO_ROOT / "src").rglob("*.py"):
        for match in _ENV_READ.finditer(path.read_text(encoding="utf-8")):
            found.add(match.group(1) or match.group(2))
    return found


def _script_files() -> list[Path]:
    return sorted(SCRIPTS_DIR.glob("*.sh"))


def test_env_example_documents_every_directly_read_var():
    documented = _env_example_keys()
    missing = sorted(v for v in _directly_read_env_vars() if v not in documented)
    assert not missing, f".env.example is missing env vars read by the code: {missing}"


def test_pyproject_dependency_hygiene():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"])
    dev = " ".join(data["project"].get("optional-dependencies", {}).get("dev", []))
    # Dead deps removed.
    assert "docker" not in deps, "docker is unused (swebench shells out); drop it"
    assert "pytest-asyncio" not in dev, "pytest-asyncio is unused; drop it"
    # Real deps present (both are imported at module load).
    assert "swebench" in deps, "swebench is imported by the test backend; declare it"
    assert "pyyaml" in deps, "skill_evolver imports yaml; declare pyyaml"
    # Bare `pytest tests -q` must work on a fresh clone.
    assert data["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_gitignore_rescues_evolved_skill_state():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()]
    assert "*.json" in lines, "the blanket *.json rule should still exist"
    # The negation must come after *.json so it wins (last match).
    assert "!skills/metadata.json" in lines
    assert lines.index("!skills/metadata.json") > lines.index("*.json")
    # `/benchmarks` (anchored, no trailing slash) also ignores the symlink, not
    # just a real directory.
    assert "*.sif" in lines and "/benchmarks" in lines


def test_license_is_mit():
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text


def test_scripts_are_portable_and_strict():
    scripts = _script_files()
    assert scripts, "no experiment scripts found"
    abs_path = re.compile(r"(^|[\s'\"=(])/(?:root|home|shared|Users|tmp)/")
    for script in scripts:
        body = script.read_text(encoding="utf-8")
        first = body.splitlines()[0]
        assert first.startswith("#!") and "bash" in first, f"{script.name}: missing bash shebang"
        assert not abs_path.search(body), f"{script.name}: contains an absolute path"
        if script.name == "common.sh":
            # The sourced library derives REPO_ROOT from its own location.
            assert 'REPO_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"' in body
            continue
        assert "set -euo pipefail" in body, f"{script.name}: missing strict mode"
        assert "common.sh" in body, f"{script.name}: does not source common.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_scripts_pass_bash_syntax_check():
    for script in _script_files():
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{script.name}: bash -n failed:\n{result.stderr}"
