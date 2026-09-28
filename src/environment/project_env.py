"""Git repository environment: file access, patch application, and test runs."""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from git import Repo
from git.exc import GitCommandError

from src.environment.models import Issue, PatchApplyResult, RepoState, TestResult

logger = logging.getLogger(__name__)


class ProjectEnvironment:
    def __init__(
        self,
        repo_path: str | Path,
        test_cmd: Optional[str] = None,
        timeout: int = 300,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.test_cmd = test_cmd or self._detect_test_cmd()
        self.timeout = timeout
        self._repo: Optional[Repo] = None
        self._original_commit: Optional[str] = None
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
    
    @property
    def repo(self) -> Repo:
        if self._repo is None:
            self._repo = Repo(self.repo_path)
        return self._repo
    
    def _detect_test_cmd(self) -> str:
        if (self.repo_path / "pytest.ini").exists():
            return "pytest"
        if (self.repo_path / "setup.py").exists():
            return "python -m pytest"
        if (self.repo_path / "pyproject.toml").exists():
            return "pytest"
        if (self.repo_path / "tox.ini").exists():
            return "tox"
        return "pytest"
    
    def get_repo_state(self) -> RepoState:
        try:
            current_branch = self.repo.active_branch.name
        except TypeError:
            # Detached HEAD.
            current_branch = "HEAD"
        
        current_commit = self.repo.head.commit.hexsha
        is_dirty = self.repo.is_dirty()
        
        modified_files = []
        if is_dirty:
            modified_files = [item.a_path for item in self.repo.index.diff(None)]
            modified_files.extend([item.a_path for item in self.repo.index.diff("HEAD")])
        
        return RepoState(
            path=self.repo_path,
            current_branch=current_branch,
            current_commit=current_commit,
            is_dirty=is_dirty,
            modified_files=list(set(modified_files)),
        )
    
    def get_file_content(self, file_path: str) -> str:
        full_path = self.repo_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return full_path.read_text(encoding="utf-8")
    
    def get_file_content_with_lines(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> str:
        """Return lines ``start_line..end_line`` (1-indexed, inclusive)."""
        content = self.get_file_content(file_path)
        lines = content.split("\n")
        
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        
        return "\n".join(lines[start_idx:end_idx])
    
    def list_files(self, pattern: str = "**/*.py") -> List[str]:
        files = []
        for path in self.repo_path.glob(pattern):
            if path.is_file() and ".git" not in str(path):
                files.append(str(path.relative_to(self.repo_path)))
        return sorted(files)
    
    @staticmethod
    def _fix_hunk_counts(patch_content: str) -> str:
        """Recompute hunk-header line counts, which LLMs frequently get wrong."""
        import re
        hunk_header = re.compile(
            r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)'
        )
        output_lines: list[str] = []
        i = 0
        lines = patch_content.splitlines(keepends=True)
        while i < len(lines):
            line = lines[i]
            m = hunk_header.match(line)
            if m:
                old_start = m.group(1)
                new_start = m.group(2)
                hint = m.group(3)
                hunk_lines: list[str] = []
                i += 1
                while i < len(lines) and not hunk_header.match(lines[i]) \
                        and not lines[i].startswith('--- ') \
                        and not lines[i].startswith('+++ '):
                    hunk_lines.append(lines[i])
                    i += 1
                old_count = sum(
                    1 for l in hunk_lines if l.startswith(' ') or l.startswith('-')
                )
                new_count = sum(
                    1 for l in hunk_lines if l.startswith(' ') or l.startswith('+')
                )
                fixed_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{hint}\n"
                output_lines.append(fixed_header)
                output_lines.extend(hunk_lines)
            else:
                output_lines.append(line)
                i += 1
        return "".join(output_lines)

    def apply_patch_detailed(self, patch_content: str) -> PatchApplyResult:
        """Apply a diff via git apply, then git apply with fixed hunk counts, then patch --fuzz=5."""
        if not patch_content.strip():
            logger.warning("Empty patch content, nothing to apply")
            return PatchApplyResult(
                success=False,
                strategy="empty",
                error_message="Empty patch content",
            )

        if self._original_commit is None:
            self._original_commit = self.repo.head.commit.hexsha

        attempts: list[dict[str, str]] = []
        patch_file = None
        fixed_patch_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(patch_content)
                patch_file = f.name

            try:
                stdout = self.repo.git.apply(patch_file, "--ignore-whitespace", "--verbose")
                logger.info("Patch applied successfully (git apply)")
                return PatchApplyResult(
                    success=True,
                    strategy="git_apply",
                    stdout=stdout or "",
                    attempts=attempts,
                )
            except GitCommandError as e:
                attempts.append({
                    "strategy": "git_apply",
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or str(e),
                })
                logger.warning(
                    f"git apply failed, trying fixed hunk counts: {(e.stderr or str(e))[:120]}"
                )

            fixed_content = self._fix_hunk_counts(patch_content)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(fixed_content)
                fixed_patch_file = f.name

            try:
                stdout = self.repo.git.apply(
                    fixed_patch_file, "--ignore-whitespace", "--verbose"
                )
                logger.info("Patch applied successfully (git apply + fixed hunk counts)")
                return PatchApplyResult(
                    success=True,
                    strategy="git_apply_fixed_hunk_counts",
                    stdout=stdout or "",
                    fixed_patch_content=fixed_content,
                    attempts=attempts,
                )
            except GitCommandError as e:
                attempts.append({
                    "strategy": "git_apply_fixed_hunk_counts",
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or str(e),
                })
                logger.warning(
                    "git apply with fixed counts failed, trying patch command: "
                    f"{(e.stderr or str(e))[:120]}"
                )

            try:
                result = subprocess.run(
                    [
                        "patch", "-p1",
                        "--fuzz=5",
                        "--ignore-whitespace",
                        "-i", fixed_patch_file,
                    ],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                attempts.append({
                    "strategy": "patch_fuzz",
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                })
                if result.returncode == 0:
                    logger.info("Patch applied successfully (patch --fuzz=5)")
                    return PatchApplyResult(
                        success=True,
                        strategy="patch_fuzz",
                        stdout=result.stdout or "",
                        stderr=result.stderr or "",
                        fixed_patch_content=fixed_content,
                        attempts=attempts,
                    )
                logger.error(f"patch --fuzz=5 failed: {result.stderr[:200]}")
                return PatchApplyResult(
                    success=False,
                    strategy="failed",
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    error_message="All patch application strategies failed",
                    fixed_patch_content=fixed_content,
                    attempts=attempts,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.error(f"patch fallback failed: {e}")
                attempts.append({
                    "strategy": "patch_fuzz",
                    "stdout": "",
                    "stderr": str(e),
                })
                return PatchApplyResult(
                    success=False,
                    strategy="patch_fuzz_error",
                    error_message=str(e),
                    fixed_patch_content=fixed_content,
                    attempts=attempts,
                )

        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            return PatchApplyResult(
                success=False,
                strategy="error",
                error_message=str(e),
                attempts=attempts,
            )
        finally:
            for fp in (patch_file, fixed_patch_file):
                if fp:
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass

    def revert_changes(self) -> bool:
        try:
            self.repo.git.checkout("--", ".")
            self.repo.git.clean("-fd")
            logger.info("Changes reverted successfully")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to revert changes: {e}")
            return False
    
    def reset_to_commit(self, commit: Optional[str] = None) -> bool:
        """Hard-reset to ``commit`` (default: the original commit)."""
        target = commit or self._original_commit
        if not target:
            logger.warning("No commit specified and no original commit recorded")
            return self.revert_changes()
        
        try:
            self.repo.git.reset("--hard", target)
            self.repo.git.clean("-fd")
            logger.info(f"Reset to commit {target[:8]}")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to reset: {e}")
            return False
    
    def run_tests(
        self,
        test_cmd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> TestResult:
        cmd = test_cmd or self.test_cmd
        timeout = timeout or self.timeout

        logger.info(f"Running tests: {cmd}")

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(self.repo_path)},
            )

            passed = result.returncode == 0

            logger.info(f"Tests {'passed' if passed else 'failed'}")

            return TestResult.from_pytest_output(
                passed=passed,
                output=result.stdout,
                error_logs=result.stderr,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Test execution timed out after {timeout}s")
            return TestResult(
                passed=False,
                output="",
                error_logs=f"Test execution timed out after {timeout} seconds",
            )
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return TestResult(
                passed=False,
                output="",
                error_logs=str(e),
            )
    
    def get_diff(self) -> str:
        """Unstaged diff plus untracked files (the staged test patch is excluded)."""
        try:
            tracked = self.repo.git.diff()
        except GitCommandError:
            return ""
        # Append untracked files without touching the index, which holds the
        # staged SWE-bench test patch.
        parts = [tracked] if tracked.strip() else []
        try:
            untracked = self.repo.git.ls_files("--others", "--exclude-standard").splitlines()
        except GitCommandError:
            untracked = []
        for path in sorted(item for item in untracked if item.strip()):
            # ``--no-index`` exits 1 whenever the files differ (always, here).
            result = subprocess.run(
                ["git", "diff", "--no-index", "--", "/dev/null", path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            text = result.stdout.strip()
            if result.returncode in (0, 1) and text and "Binary files" not in text:
                parts.append(text)
        return "\n".join(parts) + ("\n" if parts else "")
    
    def checkout_commit(self, commit: str) -> bool:
        try:
            self.repo.git.checkout(commit)
            self._original_commit = commit
            logger.info(f"Checked out commit {commit[:8]}")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to checkout {commit}: {e}")
            return False
    
    def clone_repo(self, url: str, branch: Optional[str] = None) -> bool:
        try:
            if self.repo_path.exists():
                import shutil
                shutil.rmtree(self.repo_path)
            
            self._repo = Repo.clone_from(url, self.repo_path)
            
            if branch:
                self.repo.git.checkout(branch)
            
            self._original_commit = self.repo.head.commit.hexsha
            logger.info(f"Cloned {url} to {self.repo_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            return False
    
    def setup_issue(self, issue: Issue, apply_test_patch: bool = True) -> bool:
        """Check out the issue's base commit; ``apply_test_patch`` is for host-side tests only (it exposes hidden F2P tests)."""
        # Benchmark episodes start clean: a previous staged test patch survives
        # ``checkout -- .``. A plain `fix` run on a user repo must keep local work.
        if issue.base_commit or issue.test_patch:
            try:
                self.repo.git.reset("--hard")
                self.repo.git.clean("-fd")
            except GitCommandError as e:
                logger.error(f"Failed to clean repository before setup: {e}")
                return False

        if issue.base_commit:
            if not self.checkout_commit(issue.base_commit):
                return False
        
        if issue.test_patch and apply_test_patch:
            test_patch_result = self.apply_patch_detailed(issue.test_patch)
            if not test_patch_result.success:
                logger.error(
                    f"Failed to apply test patch: {test_patch_result.diagnostic[:500]}"
                )
                self.revert_changes()
                return False
            # Stage the test patch so `git diff` (the prediction) excludes it.
            try:
                self.repo.git.add("-A")
            except GitCommandError as e:
                logger.warning(f"Failed to stage test patch: {e}")

        return True
