"""
Single-agent zero-shot baseline.

One LLM call sees the issue, the repository file tree, and the full text of the
few files with the highest lexical overlap with the issue, and returns a unified
diff. The diff is canonicalized through the ProjectEnvironment (apply -> git diff
-> revert). There are no worker roles, retries, judge, or controller guidance —
this is the reference point the multi-agent loop is compared against.

``run_single_agent`` returns the same ``ExecutionResult`` the orchestrator
produces, so the benchmark runner can dispatch to it for ``--agent-mode single``
and reuse the downstream grading, reward, and rollout-logging path unchanged.
"""

import logging
import re
import time
from typing import List, Optional

from src.environment.models import ExecutionContext, Issue, PatchInfo
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.orchestrator.orchestrator import ExecutionResult, ExecutionStatus, IterationRecord
from src.workers.base import BaseWorker, WorkerResult
from src.workers.patch_generator import PatchResult, parse_patch_response

logger = logging.getLogger(__name__)

MAX_TREE_FILES = 100
TOP_FILES = 3
SHORTLIST_FILES = 40
MAX_FILE_CHARS = 6000
CONTENT_SCAN_CHARS = 20000

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "when", "will",
    "would", "should", "could", "have", "has", "had", "not", "are", "was",
    "were", "been", "but", "into", "does", "using", "use", "code", "test",
    "tests", "error", "issue", "python", "file", "files", "line", "lines",
    "self", "none", "true", "false", "return", "import", "class", "def",
    "function", "value", "values", "which", "there", "their", "then", "than",
})


SINGLE_AGENT_SYSTEM_PROMPT = """You are an expert software engineer. You are given a bug report, the repository
file tree, and the most relevant source files. Fix the bug in one shot.

Respond with a single unified diff (git diff style) and nothing else, inside a
fenced code block:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,6 +10,6 @@
 context line
-old line
+new line
 context line
```

Rules:
- Make the smallest change that fixes the issue.
- Every hunk body line starts with exactly one prefix: ' ' (context), '-', or '+'.
- Use real repository paths from the provided files.
- Include a few context lines around each change."""


def _keywords(text: str) -> set:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")} - _STOPWORDS


class SingleAgent(BaseWorker):
    """One-shot patch generator over issue text and lexically relevant files."""

    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(llm_client=llm_client, name="SingleAgent")
        self.env = env

    @property
    def system_prompt(self) -> str:
        return SINGLE_AGENT_SYSTEM_PROMPT

    def execute(self, context: ExecutionContext) -> WorkerResult[PatchResult]:
        """Generate a patch from a single LLM call."""
        self.logger.info(f"Single-agent patch for issue: {context.issue.id}")
        try:
            prompt = self._build_prompt(context)
            response = self._call_llm(prompt)
            result = parse_patch_response(response.content)
            if result.patch_content:
                result.patch_info = PatchInfo.from_diff(result.patch_content)
            return WorkerResult(success=True, data=result, llm_response=response)
        except Exception as e:
            self.logger.error(f"Single-agent generation failed: {e}")
            return WorkerResult(success=False, error=str(e))

    def _build_prompt(self, context: ExecutionContext) -> str:
        issue = context.issue
        parts = ["## Issue Description", issue.description]
        if issue.hints:
            parts.extend(["\n## Hints", issue.hints])

        parts.append(f"\n## Repository: {context.repo_state.path.name}")
        try:
            files = self.env.list_files("**/*.py")
        except Exception as e:
            self.logger.warning(f"Could not list files: {e}")
            files = []

        if files:
            parts.append("\n## Python Files in Repository:")
            parts.append("\n".join(f"- {f}" for f in files[:MAX_TREE_FILES]))

        top_files = self._top_files(f"{issue.description}\n{issue.hints or ''}", files)
        if top_files:
            parts.append("\n## Most Relevant Files:")
            for file_path in top_files:
                try:
                    content = self.env.get_file_content(file_path)
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
                    continue
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
                parts.append(f"\n### {file_path}")
                parts.append(f"```python\n{content}\n```")

        parts.append("\n## Task")
        parts.append(
            "Return a single unified diff that fixes the issue, inside one ```diff block."
        )
        return "\n".join(parts)

    def _top_files(self, text: str, files: List[str]) -> List[str]:
        """Rank files by lexical overlap of their path and content with the issue."""
        keywords = _keywords(text)
        if not files:
            return []
        if not keywords:
            return files[:TOP_FILES]

        path_scores = {f: len(keywords & _keywords(f)) for f in files}
        shortlist = sorted(files, key=lambda f: (-path_scores[f], f))[:SHORTLIST_FILES]

        scores = {}
        for file_path in shortlist:
            score = path_scores[file_path] * 2
            try:
                content = self.env.get_file_content(file_path)[:CONTENT_SCAN_CHARS].lower()
                score += sum(1 for keyword in keywords if keyword in content)
            except Exception:
                pass
            scores[file_path] = score
        return sorted(shortlist, key=lambda f: (-scores[f], f))[:TOP_FILES]


def run_single_agent(
    issue: Issue,
    env: ProjectEnvironment,
    llm_client: Optional[LLMClient] = None,
) -> ExecutionResult:
    """Run the single-agent baseline and return an ExecutionResult.

    The baseline does not run tests; ``success`` means a non-empty canonical
    patch (a prediction) was produced. Official resolution is graded downstream
    by the test backend / SWE-bench harness.
    """
    start_time = time.time()
    agent = SingleAgent(env, llm_client)
    repo_state = env.get_repo_state()
    context = ExecutionContext(issue=issue, repo_state=repo_state, max_iterations=1)

    if not env.setup_issue(issue):
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            issue_id=issue.id,
            error_message="Failed to set up issue environment",
        )

    record = IterationRecord(iteration=0)
    worker_result = agent.execute(context)
    record.patch_result = worker_result.data
    record.tokens_used = worker_result.tokens_used

    final_patch: Optional[PatchInfo] = None
    patch_content = worker_result.data.patch_content if worker_result.data else ""
    if patch_content.strip():
        apply_result = env.apply_patch_detailed(patch_content)
        if apply_result.success:
            canonical_diff = env.get_diff()
            if canonical_diff.strip():
                final_patch = PatchInfo.from_diff(canonical_diff)
            env.revert_changes()

    record.duration_ms = (time.time() - start_time) * 1000
    success = final_patch is not None
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
        issue_id=issue.id,
        success=success,
        iterations_used=1,
        total_tokens=record.tokens_used,
        total_duration_ms=record.duration_ms,
        final_patch=final_patch,
        iteration_records=[record],
        metadata={"agent_mode": "single"},
    )
