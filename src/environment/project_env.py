"""
Project Environment for code repository interaction.

Provides an interface for interacting with code repositories,
running tests, and applying patches.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from git import Repo
from git.exc import GitCommandError

from src.environment.models import Issue, PatchInfo, RepoState, TestResult

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
    ) -> str:
        """
        Read specific lines from a file.
        
        Args:
            file_path: Relative path to the file.
            start_line: Starting line number (1-indexed).
            end_line: Ending line number (inclusive). None means end of file.
            
        Returns:
            Selected lines as string.
        """
        content = self.get_file_content(file_path)
        lines = content.split("\n")
        
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        
        return "\n".join(lines[start_idx:end_idx])
    
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
    
    @staticmethod
    def _fix_hunk_counts(patch_content: str) -> str:
        """
        Recompute and fix the line counts in unified diff hunk headers.

        LLMs frequently generate wrong @@ -X,Y +A,B @@ counts.
        This function parses each hunk, counts the actual old/new lines,
        and rewrites the headers so that tools like git-apply accept them.
        """
        import re
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

    def apply_patch(self, patch_content: str) -> bool:
        """
        Apply a unified diff patch to the repository.

        Attempts multiple strategies in order:
        1. git apply --ignore-whitespace (standard)
        2. git apply with fixed hunk counts + --ignore-whitespace
        3. patch -p1 --fuzz=5 --ignore-whitespace (most lenient)

        Args:
            patch_content: The patch in unified diff format.

        Returns:
            True if patch was applied successfully, False otherwise.
        """
        if not patch_content.strip():
            logger.warning("Empty patch content, nothing to apply")
            return False

        # Save original commit for potential rollback
        if self._original_commit is None:
            self._original_commit = self.repo.head.commit.hexsha

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
                self.repo.git.apply(patch_file, "--ignore-whitespace", "--verbose")
                logger.info("Patch applied successfully (git apply)")
                return True
            except GitCommandError as e:
                logger.warning(f"git apply failed, trying fixed hunk counts: {e.stderr[:120]}")

            # Strategy 2: fix hunk counts then git apply
            fixed_content = self._fix_hunk_counts(patch_content)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False, encoding="utf-8"
            ) as f:
                f.write(fixed_content)
                fixed_patch_file = f.name

            try:
                self.repo.git.apply(
                    fixed_patch_file, "--ignore-whitespace", "--verbose"
                )
                logger.info("Patch applied successfully (git apply + fixed hunk counts)")
                return True
            except GitCommandError as e:
                logger.warning(
                    f"git apply with fixed counts failed, trying patch command: {e.stderr[:120]}"
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
                if result.returncode == 0:
                    logger.info("Patch applied successfully (patch --fuzz=5)")
                    return True
                logger.error(f"patch --fuzz=5 failed: {result.stderr[:200]}")
                return False
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.error(f"patch fallback failed: {e}")
                return False

        except Exception as e:
            logger.error(f"Error applying patch: {e}")
            return False
        finally:
            for fp in (patch_file, fixed_patch_file):
                if fp:
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass
    
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
            if not self.apply_patch(issue.test_patch):
                logger.warning("Failed to apply test patch")
        
        return True
