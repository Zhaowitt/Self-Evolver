"""
Data models for Project Environment.

Defines the core data structures used across the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_patch_text(patch_content: str) -> str:
    """Normalize patch line endings and ensure a final newline."""
    patch_content = patch_content.replace("\r\n", "\n").replace("\r", "\n")
    if patch_content and not patch_content.endswith("\n"):
        patch_content += "\n"
    return patch_content


class TestStatus(Enum):
    """Status of a test execution."""
    
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class CodeLocation:
    """A specific location in source code."""
    
    file_path: str
    start_line: int
    end_line: Optional[int] = None
    snippet: Optional[str] = None
    
    def __str__(self) -> str:
        if self.end_line and self.end_line != self.start_line:
            return f"{self.file_path}:{self.start_line}-{self.end_line}"
        return f"{self.file_path}:{self.start_line}"


@dataclass
class Issue:
    """Represents a coding task or issue to be solved."""
    
    id: str
    description: str
    repo_name: Optional[str] = None
    base_commit: Optional[str] = None
    hints: Optional[str] = None
    test_patch: Optional[str] = None  # For SWE-bench: the test to verify fix
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Issue({self.id}): {self.description[:100]}..."


@dataclass
class TestCase:
    """A single test case result."""
    
    name: str
    status: TestStatus
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
    
    @property
    def failed(self) -> bool:
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)


@dataclass
class TestResult:
    """Result of running tests."""
    
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    test_cases: List[TestCase] = field(default_factory=list)
    output: str = ""
    error_logs: str = ""
    duration_ms: float = 0.0
    
    @property
    def failed_test_names(self) -> List[str]:
        """Get names of failed tests."""
        return [tc.name for tc in self.test_cases if tc.failed]
    
    @property
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"Tests: {self.passed_tests}/{self.total_tests} passed, "
            f"{self.failed_tests} failed, {self.error_tests} errors"
        )
    
    @classmethod
    def from_success(cls, output: str = "") -> "TestResult":
        """Create a successful test result."""
        return cls(passed=True, output=output)
    
    @classmethod
    def from_failure(cls, error_logs: str, output: str = "") -> "TestResult":
        """Create a failed test result."""
        return cls(passed=False, error_logs=error_logs, output=output)


@dataclass
class RepoState:
    """Current state of a code repository."""
    
    path: Path
    current_branch: str = "main"
    current_commit: Optional[str] = None
    is_dirty: bool = False
    modified_files: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "dirty" if self.is_dirty else "clean"
        return f"Repo({self.path.name}@{self.current_branch}, {status})"


@dataclass
class PatchInfo:
    """Information about a code patch."""
    
    content: str  # Unified diff format
    modified_files: List[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0
    
    @property
    def total_changes(self) -> int:
        return self.added_lines + self.removed_lines
    
    @classmethod
    def from_diff(cls, diff_content: str) -> "PatchInfo":
        """Parse patch info from unified diff content."""
        diff_content = normalize_patch_text(diff_content)
        modified_files = []
        added = 0
        removed = 0
        
        for line in diff_content.split("\n"):
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                file_path = line[6:]  # Remove "--- a/" or "+++ b/"
                if file_path not in modified_files and file_path != "/dev/null":
                    modified_files.append(file_path)
            elif line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        
        return cls(
            content=diff_content,
            modified_files=modified_files,
            added_lines=added,
            removed_lines=removed,
        )


@dataclass
class PatchApplyResult:
    """Structured result from attempting to apply a patch."""

    success: bool
    strategy: str = ""
    stdout: str = ""
    stderr: str = ""
    error_message: str = ""
    fixed_patch_content: Optional[str] = None
    attempts: List[Dict[str, str]] = field(default_factory=list)

    @property
    def diagnostic(self) -> str:
        """Compact diagnostic string for retry prompts and logging."""
        parts = []
        if self.error_message:
            parts.append(self.error_message)
        if self.stderr:
            parts.append(self.stderr)
        if self.stdout:
            parts.append(self.stdout)
        return "\n".join(part for part in parts if part).strip()


@dataclass
class PatchContextCheckResult:
    """Result of checking whether patch hunk context matches repository files."""

    success: bool
    diagnostic: str = ""
    file_path: str = ""
    hunk_header: str = ""
    expected_line: str = ""
    actual_line: str = ""


@dataclass
class ExecutionContext:
    """Context passed between workers during execution."""
    
    issue: Issue
    repo_state: RepoState
    iteration: int = 0
    max_iterations: int = 3
    previous_patches: List[PatchInfo] = field(default_factory=list)
    previous_errors: List[str] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_previous_attempt(self) -> bool:
        return self.iteration > 0
    
    @property
    def last_error(self) -> Optional[str]:
        return self.previous_errors[-1] if self.previous_errors else None
    
    @property
    def last_test_result(self) -> Optional[TestResult]:
        return self.test_results[-1] if self.test_results else None
