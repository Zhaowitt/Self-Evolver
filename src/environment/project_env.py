"""
Project Environment for code repository interaction.

Provides an interface for interacting with code repositories,
running tests, and applying patches.
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from git import Repo
from git.exc import GitCommandError

from src.environment.models import (
    Issue,
    PatchApplyResult,
    PatchContextCheckResult,
    RepoState,
    TestResult,
    normalize_patch_text,
)

logger = logging.getLogger(__name__)


class ProjectEnvironment:
    """
    Manages interaction with a code project environment.
    
    Provides methods for:
    - Reading repository state and file contents
    - Applying and reverting patches
    - Running tests
    - Managing git operations
    """
    
    def __init__(
        self,
        repo_path: str | Path,
        test_cmd: Optional[str] = None,
        timeout: int = 300,
    ):
        """
        Initialize the project environment.
        
        Args:
            repo_path: Path to the repository root.
            test_cmd: Command to run tests (e.g., "pytest", "python -m pytest").
            timeout: Timeout in seconds for test execution.
        """
        self.repo_path = Path(repo_path).resolve()
        self.test_cmd = test_cmd or self._detect_test_cmd()
        self.timeout = timeout
        self._repo: Optional[Repo] = None
        self._original_commit: Optional[str] = None
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
    
    @property
    def repo(self) -> Repo:
        """Get the git repository object."""
        if self._repo is None:
            self._repo = Repo(self.repo_path)
        return self._repo
    
    def _detect_test_cmd(self) -> str:
        """Auto-detect the test command based on project files."""
        if (self.repo_path / "pytest.ini").exists():
            return "pytest"
        if (self.repo_path / "setup.py").exists():
            return "python -m pytest"
        if (self.repo_path / "pyproject.toml").exists():
            return "pytest"
        if (self.repo_path / "tox.ini").exists():
            return "tox"
        return "pytest"  # Default fallback
    
    def get_repo_state(self) -> RepoState:
        """Get the current state of the repository."""
        try:
            current_branch = self.repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            current_branch = "HEAD"
        
        current_commit = self.repo.head.commit.hexsha
        is_dirty = self.repo.is_dirty()
        
        # Get list of modified files
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
        """
        Read the content of a file in the repository.
        
        Args:
            file_path: Relative path to the file from repo root.
            
        Returns:
            File content as string.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        full_path = self.repo_path / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return full_path.read_text(encoding="utf-8")
    
    def get_file_content_with_lines(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        include_line_numbers: bool = False,
    ) -> str:
        """
        Read specific lines from a file.
        
        Args:
            file_path: Relative path to the file.
            start_line: Starting line number (1-indexed).
            end_line: Ending line number (inclusive). None means end of file.
            include_line_numbers: Prefix returned lines with 1-indexed line numbers.
            
        Returns:
            Selected lines as string.
        """
        content = self.get_file_content(file_path)
        lines = content.split("\n")
        
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        selected = lines[start_idx:end_idx]

        if include_line_numbers:
            return "\n".join(
                f"{start_idx + offset + 1:5d}: {line}"
                for offset, line in enumerate(selected)
            )
        
        return "\n".join(selected)
    
    def list_files(self, pattern: str = "**/*.py") -> List[str]:
        """
        List files matching a pattern.
        
        Args:
            pattern: Glob pattern for matching files.
            
        Returns:
            List of relative file paths.
        """
        files = []
        for path in self.repo_path.glob(pattern):
            if path.is_file() and ".git" not in str(path):
                files.append(str(path.relative_to(self.repo_path)))
        return sorted(files)

    def check_patch_context(self, patch_content: str) -> PatchContextCheckResult:
        """
        Check that each unified diff hunk's old-side context matches files.

        This catches LLM-generated hunks whose context is stale, truncated, or
        fabricated before lenient patch application can place them incorrectly.
        """
        patch_content = normalize_patch_text(patch_content)
        hunk_header = re.compile(
            r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
            r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
        )
        lines = patch_content.splitlines()
        old_file_path = ""
        file_path = ""
        i = 0

        while i < len(lines):
            line = lines[i]
            if line.startswith("--- "):
                old_file_path = self._diff_path_to_repo_path(line[4:].strip())
                i += 1
                continue
            if line.startswith("+++ "):
                file_path = self._diff_path_to_repo_path(line[4:].strip()) or old_file_path
                i += 1
                continue

            match = hunk_header.match(line)
            if not match:
                i += 1
                continue

            header = line
            if not file_path:
                return PatchContextCheckResult(
                    success=False,
                    diagnostic=f"Hunk has no target file header: {header}",
                    hunk_header=header,
                )

            old_start = int(match.group("old_start"))
            old_count = int(match.group("old_count") or "1")
            old_lines: list[str] = []
            i += 1

            while i < len(lines) and not hunk_header.match(lines[i]) \
                    and not lines[i].startswith("diff --git ") \
                    and not lines[i].startswith("--- ") \
                    and not lines[i].startswith("+++ "):
                hunk_line = lines[i]
                if hunk_line.startswith("\\"):
                    i += 1
                    continue
                if not hunk_line:
                    return PatchContextCheckResult(
                        success=False,
                        diagnostic=(
                            "Malformed hunk body line without unified diff prefix "
                            f"in {file_path}: {header}"
                        ),
                        file_path=file_path,
                        hunk_header=header,
                    )
                prefix = hunk_line[0]
                if prefix in {" ", "-"}:
                    old_lines.append(hunk_line[1:])
                elif prefix != "+":
                    return PatchContextCheckResult(
                        success=False,
                        diagnostic=(
                            f"Invalid hunk line prefix {prefix!r} in "
                            f"{file_path}: {header}"
                        ),
                        file_path=file_path,
                        hunk_header=header,
                    )
                i += 1

            if old_count == 0 and not old_lines:
                continue

            try:
                actual_file_lines = self.get_file_content(file_path).splitlines()
            except FileNotFoundError:
                return PatchContextCheckResult(
                    success=False,
                    diagnostic=f"Patch references missing file: {file_path}",
                    file_path=file_path,
                    hunk_header=header,
                )

            start_idx = max(0, old_start - 1)
            actual_lines = actual_file_lines[start_idx:start_idx + len(old_lines)]
            if actual_lines != old_lines:
                mismatch_idx = self._first_mismatch_index(old_lines, actual_lines)
                expected = old_lines[mismatch_idx] if mismatch_idx < len(old_lines) else ""
                actual = actual_lines[mismatch_idx] if mismatch_idx < len(actual_lines) else "<EOF>"
                line_no = old_start + mismatch_idx
                return PatchContextCheckResult(
                    success=False,
                    diagnostic=(
                        f"Patch context mismatch in {file_path} at line {line_no}. "
                        f"Hunk: {header}. Expected old-side line {expected!r}, "
                        f"actual file line is {actual!r}."
                    ),
                    file_path=file_path,
                    hunk_header=header,
                    expected_line=expected,
                    actual_line=actual,
                )

        return PatchContextCheckResult(success=True)

    @staticmethod
    def _diff_path_to_repo_path(diff_path: str) -> str:
        if diff_path in {"/dev/null", "dev/null"}:
            return ""
        if diff_path.startswith("a/") or diff_path.startswith("b/"):
            return diff_path[2:]
        return diff_path

    @staticmethod
    def _first_mismatch_index(expected: list[str], actual: list[str]) -> int:
        for idx, expected_line in enumerate(expected):
            if idx >= len(actual) or actual[idx] != expected_line:
                return idx
        return len(expected)
    
    @staticmethod
    def _fix_hunk_counts(patch_content: str) -> str:
        """
        Recompute and fix the line counts in unified diff hunk headers.

        LLMs frequently generate wrong @@ -X,Y +A,B @@ counts.
        This function parses each hunk, counts the actual old/new lines,
        and rewrites the headers so that tools like git-apply accept them.
        """
        patch_content = normalize_patch_text(patch_content)
        # Capture: old_start, new_start, and optional trailing function hint
        # e.g.  "@@ -78,6 +78,15 @@ def foo" -> groups: "78", "78", " @@ def foo"
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
                hint = m.group(3)  # e.g. "" or " def foo"
                # Collect all lines belonging to this hunk
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
        """
        Apply a unified diff patch to the repository and return diagnostics.

        Attempts multiple strategies in order:
        1. git apply --ignore-whitespace (standard)
        2. git apply with fixed hunk counts + --ignore-whitespace
        3. patch -p1 --fuzz=5 --ignore-whitespace (most lenient)

        Args:
            patch_content: The patch in unified diff format.

        Returns:
            PatchApplyResult with success state, strategy, and command diagnostics.
        """
        patch_content = normalize_patch_text(patch_content)

        if not patch_content.strip():
            logger.warning("Empty patch content, nothing to apply")
            return PatchApplyResult(
                success=False,
                strategy="empty",
                error_message="Empty patch content",
            )

        # Save original commit for potential rollback
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

            # Strategy 1: git apply --ignore-whitespace
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

            # Strategy 2: fix hunk counts then git apply
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

            # Strategy 3: patch --fuzz=5 --ignore-whitespace (most lenient)
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

    def apply_patch(self, patch_content: str) -> bool:
        """
        Apply a unified diff patch to the repository.

        Kept as a bool-returning compatibility wrapper around
        apply_patch_detailed().
        """
        return self.apply_patch_detailed(patch_content).success
    
    def revert_changes(self) -> bool:
        """
        Revert all changes in the repository.
        
        Returns:
            True if revert was successful.
        """
        try:
            self.repo.git.checkout("--", ".")
            self.repo.git.clean("-fd")
            logger.info("Changes reverted successfully")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to revert changes: {e}")
            return False
    
    def reset_to_commit(self, commit: Optional[str] = None) -> bool:
        """
        Reset repository to a specific commit.
        
        Args:
            commit: Commit hash to reset to. Uses original commit if None.
            
        Returns:
            True if reset was successful.
        """
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
        """
        Run tests in the repository.
        
        Args:
            test_cmd: Test command to run. Uses default if None.
            timeout: Timeout in seconds. Uses default if None.
            
        Returns:
            TestResult with pass/fail status and output.
        """
        cmd = test_cmd or self.test_cmd
        timeout = timeout or self.timeout
        
        logger.info(f"Running tests: {cmd}")
        
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
            
            output = result.stdout
            error_logs = result.stderr
            passed = result.returncode == 0
            
            logger.info(f"Tests {'passed' if passed else 'failed'}")
            
            return TestResult(
                passed=passed,
                output=output,
                error_logs=error_logs,
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
    
    def run_specific_tests(self, test_files: List[str]) -> TestResult:
        """
        Run specific test files.
        
        Args:
            test_files: List of test file paths to run.
            
        Returns:
            TestResult with pass/fail status.
        """
        if not test_files:
            return TestResult(passed=True, output="No tests specified")
        
        test_paths = " ".join(test_files)
        cmd = f"{self.test_cmd} {test_paths}"
        return self.run_tests(test_cmd=cmd)
    
    def get_diff(self) -> str:
        """
        Get the current diff of uncommitted changes.
        
        Returns:
            Unified diff string.
        """
        try:
            return self.repo.git.diff()
        except GitCommandError:
            return ""
    
    def checkout_commit(self, commit: str) -> bool:
        """
        Checkout a specific commit.
        
        Args:
            commit: Commit hash to checkout.
            
        Returns:
            True if checkout was successful.
        """
        try:
            self.repo.git.checkout(commit)
            self._original_commit = commit
            logger.info(f"Checked out commit {commit[:8]}")
            return True
        except GitCommandError as e:
            logger.error(f"Failed to checkout {commit}: {e}")
            return False
    
    def clone_repo(self, url: str, branch: Optional[str] = None) -> bool:
        """
        Clone a repository to the repo_path.
        
        Args:
            url: Git URL to clone from.
            branch: Branch to checkout after clone.
            
        Returns:
            True if clone was successful.
        """
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
    
    def setup_issue(self, issue: Issue) -> bool:
        """
        Set up the environment for working on an issue.
        
        Args:
            issue: The issue to work on.
            
        Returns:
            True if setup was successful.
        """
        # If base_commit is specified, checkout that commit
        if issue.base_commit:
            if not self.checkout_commit(issue.base_commit):
                return False
        
        # Apply test patch if provided (for SWE-bench)
        if issue.test_patch:
            test_patch_result = self.apply_patch_detailed(issue.test_patch)
            if not test_patch_result.success:
                logger.warning("Failed to apply test patch")
                self.revert_changes()
            else:
                # SWE-bench test patches are needed for local validation but must
                # not appear in final predictions. Staging them keeps default
                # `git diff` focused on the agent's later code changes.
                try:
                    self.repo.git.add("-A")
                except GitCommandError as e:
                    logger.warning(f"Failed to stage test patch: {e}")
        
        return True
