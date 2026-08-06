"""
Patch Generator Worker - Code Patch Generation Agent.

Responsible for generating code patches based on inspection results.
"""

import json
import logging
import re
import difflib
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from src.benchmark.datasets import source_globs
from src.controller.injection import format_controller_guidance
from src.config import get_config
from src.environment.models import ExecutionContext, PatchInfo
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient, LLMResponse, Message
from src.workers.base import BaseWorker, WorkerResult
from src.workers.inspector import InspectionResult, Inspector

logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """An exact, repository-grounded text replacement."""

    path: str
    old_text: str
    new_text: str


@dataclass
class PatchResult:
    """Result of patch generation."""
    
    patch_content: str = ""
    modified_files: List[str] = field(default_factory=list)
    explanation: str = ""
    patch_info: Optional[PatchInfo] = None
    edits: List[EditOperation] = field(default_factory=list)


PATCH_GENERATOR_SYSTEM_PROMPT = """You are an expert software engineer specializing in bug fixing and code patching.

Your task is to generate a minimal, correct patch to fix the identified bug.

## Output Format

You MUST respond with a valid JSON object containing exact text replacements:
```json
{
    "explanation": "Brief explanation of what the edits do and why",
    "edits": [
        {
            "path": "src/module.py",
            "old_text": "exact existing text copied from the provided repository context",
            "new_text": "replacement text"
        }
    ]
}
```

The program will convert these edits into a canonical unified diff. Do not write
hunk headers or a patch string yourself.

## Guidelines

1. Make MINIMAL changes - only fix what's necessary
2. Preserve existing code style and formatting
3. `old_text` must occur exactly once in the named file
4. Copy `old_text` verbatim from Relevant Code or Full File Contents, including indentation and line breaks
5. For an insertion, replace a unique nearby anchor with that anchor plus the insertion
6. Never edit tests unless the issue explicitly requires a test change
7. Prefer a small 1-10 line `old_text`; do not reconstruct code that was not shown
8. Use the repository tools to locate and read the exact target code before finalizing
9. Copy `old_text` only from the exact-content section returned by `read_file`
10. If this is a retry, use the exact repository context in the error and avoid the same mistake"""


def _normalize_patch(patch: str) -> str:
    """A unified diff must end with a newline; fenced-block extraction can drop it."""
    if patch and not patch.endswith("\n"):
        return patch + "\n"
    return patch


def _patch_result_from_json(data: Any) -> Optional[PatchResult]:
    """Build a patch result from the documented JSON object shape."""
    if not isinstance(data, dict):
        return None

    patch = data.get("patch", "")
    if not isinstance(patch, str):
        return None
    if "\n" not in patch and "\\n" in patch:
        patch = patch.replace("\\n", "\n")

    modified_files = data.get("modified_files", [])
    if not isinstance(modified_files, list):
        modified_files = []

    edits = []
    raw_edits = data.get("edits", [])
    if isinstance(raw_edits, list):
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            old_text = item.get("old_text")
            new_text = item.get("new_text")
            if all(isinstance(value, str) for value in (path, old_text, new_text)):
                edits.append(EditOperation(path=path, old_text=old_text, new_text=new_text))

    return PatchResult(
        patch_content=_normalize_patch(patch),
        modified_files=[str(path) for path in modified_files],
        explanation=str(data.get("explanation", "")),
        edits=edits,
    )


def parse_patch_response(content: str) -> PatchResult:
    """Parse a patch from an LLM response (JSON object, ```diff block, or bare diff).

    When the patch arrives as a JSON string, ``json.loads`` has already decoded
    escape sequences. Some models instead emit the whole diff on a single line
    with literal ``\\n`` separators and no real newlines; only in that case do we
    unescape, so a legitimate backslash-n inside code (e.g. ``print("\\n")``) in
    a proper multi-line diff is never turned into a real newline.
    """
    # The documented format is a JSON object. Parse that standard form first;
    # Markdown fences remain supported for models that add them anyway.
    try:
        raw_json_result = _patch_result_from_json(json.loads(content.strip()))
        if raw_json_result is not None:
            return raw_json_result
    except (json.JSONDecodeError, TypeError):
        pass

    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        try:
            fenced_json_result = _patch_result_from_json(json.loads(json_match.group(1)))
            if fenced_json_result is not None:
                return fenced_json_result
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: a fenced ```diff block, then a bare unified diff anywhere.
    diff_match = re.search(r'```(?:diff)?\s*(---.*?)\s*```', content, re.DOTALL)
    if not diff_match:
        diff_match = re.search(
            r'(--- a/.*?\n\+\+\+ b/.*?\n@@.*?(?:\n[-+ ].*)*)', content, re.DOTALL
        )
    if diff_match:
        patch = _normalize_patch(diff_match.group(1))
        return PatchResult(
            patch_content=patch,
            modified_files=PatchInfo.from_diff(patch).modified_files,
            explanation="Extracted from response",
        )

    logger.warning("Could not parse patch from response")
    return PatchResult(
        patch_content="",
        explanation=f"Failed to parse patch. Raw response: {content[:500]}",
    )


class PatchGenerator(BaseWorker):
    """Patch Generator worker for creating code fixes."""
    
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
        skill_bank: Any = None,
    ):
        super().__init__(llm_client=llm_client, name="PatchGenerator")
        self.env = env
        self.skill_bank = skill_bank
        self.max_tool_calls = max(
            0,
            get_config().agent.patch_generator_max_tool_calls,
        )
        # Reuse Inspector's single, repo-scoped implementation of the read-only
        # repository tools. The PatchGenerator owns the conversation and prompt.
        self.repo_tools = Inspector(env, self.llm_client, skill_bank=skill_bank)
    
    @property
    def system_prompt(self) -> str:
        return PATCH_GENERATOR_SYSTEM_PROMPT
    
    def execute(
        self,
        context: ExecutionContext,
        inspection_result: Optional[InspectionResult] = None,
    ) -> WorkerResult[PatchResult]:
        """Generate a patch based on inspection results."""
        self.logger.info(f"Generating patch for issue: {context.issue.id}")
        
        try:
            user_message = self._build_patch_prompt(context, inspection_result)
            additional_context = None
            if context.has_previous_attempt:
                additional_context = self._build_retry_context(context)
            
            response, tool_trace = self._call_llm_with_repo_tools(
                user_message,
                additional_context,
            )
            result = self._parse_response(response.content)

            if result.edits:
                result.patch_content = self._diff_from_edits(result.edits)
                result.modified_files = PatchInfo.from_diff(
                    result.patch_content
                ).modified_files
            
            if result.patch_content:
                result.patch_info = PatchInfo.from_diff(result.patch_content)
            
            self.logger.info(
                f"Patch generated: {len(result.modified_files)} files modified"
            )
            
            return WorkerResult(
                success=True,
                data=result,
                llm_response=response,
                metadata={"tool_trace": tool_trace},
            )
            
        except Exception as e:
            self.logger.error(f"Patch generation failed: {e}")
            return WorkerResult(success=False, error=str(e))

    def _call_llm_with_repo_tools(
        self,
        user_message: str,
        additional_context: Optional[str] = None,
    ) -> tuple[LLMResponse, List[dict[str, Any]]]:
        """Generate edits after a bounded, read-only repository inspection."""
        if additional_context:
            user_message = f"{user_message}\n\n{additional_context}"

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]
        tool_trace: List[dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tools = self.repo_tools._repo_tool_schemas()
        tool_calls_used = 0
        forced_read_prompt_sent = False

        for _ in range(self.max_tool_calls + 1):
            response = self.llm_client.chat(messages, tools=tools, tool_choice="auto")
            Inspector._accumulate_usage(total_usage, response)

            if not response.tool_calls:
                read_performed = any(
                    item["name"] == "read_file" and not item["result_preview"].startswith("ERROR:")
                    for item in tool_trace
                )
                if (
                    not read_performed
                    and self.max_tool_calls > tool_calls_used
                    and not forced_read_prompt_sent
                ):
                    forced_read_prompt_sent = True
                    messages.append(Message(role="assistant", content=response.content))
                    messages.append(Message(
                        role="user",
                        content=(
                            "Before finalizing, call read_file on the exact source file "
                            "you intend to edit. Copy old_text from the returned file "
                            "content, without the display-only line-number prefix."
                        ),
                    ))
                    continue
                response.usage = total_usage
                return response, tool_trace

            messages.append(Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            ))
            for tool_call in response.tool_calls:
                if tool_calls_used >= self.max_tool_calls:
                    result = Inspector._tool_error("Tool call budget exhausted.")
                else:
                    result = self.repo_tools._execute_repo_tool(tool_call)
                    tool_calls_used += 1
                if Inspector._tool_call_name(tool_call) == "read_file":
                    result = self._exact_read_file_result(result)
                tool_trace.append({
                    "name": Inspector._tool_call_name(tool_call),
                    "arguments": Inspector._tool_call_arguments_text(tool_call),
                    "result_preview": result[:500],
                })
                messages.append(Message(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call.get("id", ""),
                ))

            if tool_calls_used >= self.max_tool_calls:
                break

        messages.append(Message(
            role="user",
            content=(
                "Repository tool budget is exhausted. Respond now with the required "
                "JSON edits using only exact code already read. If no file was read, "
                "return an empty edits array instead of inventing code."
            ),
        ))
        final_response = self.llm_client.chat(messages)
        Inspector._accumulate_usage(total_usage, final_response)
        final_response.usage = total_usage
        return final_response, tool_trace

    @staticmethod
    def _exact_read_file_result(result: str) -> str:
        """Remove Inspector's display line numbers so copied text stays exact."""
        if result.startswith("ERROR:"):
            return result
        lines = result.splitlines()
        if not lines or not lines[0].startswith("FILE: "):
            return result
        exact_lines = [
            re.sub(r"^\s*\d+: ?", "", line)
            for line in lines[1:]
        ]
        return f"{lines[0]}\nEXACT CONTENT (copy verbatim):\n" + "\n".join(exact_lines)
    
    def _build_patch_prompt(
        self,
        context: ExecutionContext,
        inspection_result: Optional[InspectionResult],
    ) -> str:
        """Build the patch generation prompt."""
        parts = []
        
        parts.append("## Issue Description")
        parts.append(context.issue.description)

        controller_guidance = format_controller_guidance(
            context.metadata.get("controller_signal"),
            skill_bank=self.skill_bank,
        )
        if controller_guidance:
            parts.append("\n" + controller_guidance)

        # Give the smaller worker model real paths even when localization fails;
        # otherwise it may copy a schema example path into a tool call.
        try:
            language = (
                context.issue.metadata.get("repo_language")
                if context.issue.metadata else None
            )
            source_files: List[str] = []
            for glob in source_globs(language):
                source_files.extend(self.env.list_files(glob))
            source_files = list(dict.fromkeys(source_files))
            if source_files:
                parts.append("\n## Real Source Paths (use these exact paths with tools)")
                parts.append("\n".join(f"- {path}" for path in source_files[:80]))
        except Exception as e:
            self.logger.warning(f"Could not list source paths: {e}")
        
        if inspection_result:
            parts.append("\n## Fault Localization Results")
            parts.append(f"Root Cause: {inspection_result.root_cause_analysis}")
            
            if inspection_result.fix_suggestions:
                parts.append("\nSuggested Fixes:")
                for i, suggestion in enumerate(inspection_result.fix_suggestions, 1):
                    parts.append(f"{i}. {suggestion}")
            
            if inspection_result.suspected_files:
                parts.append(f"\nSuspected Files: {', '.join(inspection_result.suspected_files)}")
            
            # Include code snippets
            if inspection_result.relevant_code_snippets:
                parts.append("\n## Relevant Code:")
                for loc, code in inspection_result.relevant_code_snippets.items():
                    parts.append(f"\n### {loc}")
                    parts.append(f"```python\n{code}\n```")
        
        # Fetch full file content for suspected files
        if inspection_result and inspection_result.suspected_files:
            parts.append("\n## Full File Contents:")
            for file_path in inspection_result.suspected_files[:3]:
                try:
                    content = self.env.get_file_content(file_path)
                    if len(content) <= 12000:
                        parts.append(f"\n### {file_path}")
                        parts.append(f"```python\n{content}\n```")
                    else:
                        parts.append(
                            f"\n### {file_path}\n"
                            "(Large file: use the line-targeted Relevant Code snippet above.)"
                        )
                except FileNotFoundError:
                    self.logger.warning(f"File not found: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
        
        parts.append("\n## Task")
        parts.append(
            "Return the minimal fix as JSON exact text replacements described in the "
            "system prompt. Use read_file on every target file and only copy old_text "
            "from its EXACT CONTENT section. The JSON path above is only a schema "
            "example; never use it unless that exact file exists."
        )
        
        return "\n".join(parts)
    
    def _build_retry_context(self, context: ExecutionContext) -> str:
        """Build context for retry attempts."""
        parts = [f"## This is attempt #{context.iteration + 1}"]
        parts.append("Previous attempts have failed. Please try a different approach.")

        judge_route = context.metadata.get("next_route")
        judge_feedback = context.metadata.get("judge_feedback")
        if judge_route or judge_feedback:
            parts.append("\n## Judge Routing Feedback")
            if judge_route:
                parts.append(f"Route: {judge_route}")
            if judge_feedback:
                parts.append(str(judge_feedback))

            if judge_route == "repair_patch_format":
                parts.append(
                    "Return valid JSON with an edits array. Do not write a unified diff. "
                    "Copy each old_text exactly from the repository context in this prompt."
                )
            elif judge_route == "empty_patch_reprompt":
                parts.append(
                    "You must produce a non-empty patch. If uncertain, make the smallest "
                    "safe code change that directly addresses the issue."
                )
            elif judge_route == "regenerate_patch_same_location":
                parts.append(
                    "Keep the current localization unless the verifier logs prove it is "
                    "wrong; correct the repair logic using the latest test failure."
                )
        
        if context.previous_patches:
            parts.append("\n## Previous Patches (failed):")
            for i, patch in enumerate(context.previous_patches, 1):
                parts.append(f"\n### Patch {i}:")
                parts.append(f"```diff\n{patch.content[:2000]}\n```")

        if context.previous_errors:
            parts.append("\n## Exact Previous Errors:")
            for error in context.previous_errors[-3:]:
                parts.append(str(error)[:1500])
        
        if context.last_test_result:
            parts.append("\n## Last Test Result:")
            if context.last_test_result.error_logs:
                parts.append(f"```\n{context.last_test_result.error_logs[:1500]}\n```")
        
        parts.append("\n**Analyze why previous patches failed and generate a corrected version.**")
        return "\n".join(parts)
    
    def _parse_response(self, content: str) -> PatchResult:
        """Parse LLM response into PatchResult."""
        return parse_patch_response(content)

    def _diff_from_edits(self, edits: List[EditOperation]) -> str:
        """Apply uniquely grounded replacements and emit a program-generated diff.

        Exact matching remains the primary path. A whitespace-only fallback tolerates
        model changes to indentation, trailing spaces, or surrounding blank lines, but
        is accepted only when it identifies one repository span.
        """
        originals: dict[str, str] = {}
        updated: dict[str, str] = {}
        for edit in edits:
            path = self._validated_edit_path(edit.path)
            if not edit.old_text:
                raise ValueError(f"{path}: old_text must not be empty")
            if path not in originals:
                originals[path] = self.env.get_file_content(path)
                updated[path] = originals[path]
            updated[path] = self._apply_grounded_edit(
                path,
                updated[path],
                edit.old_text,
                edit.new_text,
            )

        chunks = []
        for path in originals:
            if originals[path] == updated[path]:
                continue
            chunks.append("".join(difflib.unified_diff(
                originals[path].splitlines(keepends=True),
                updated[path].splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )))
        patch = "".join(chunks)
        if not patch.strip():
            raise ValueError("structured edits produced no changes")
        return _normalize_patch(patch)

    def _apply_grounded_edit(
        self,
        path: str,
        content: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """Replace one exact or whitespace-equivalent repository span."""
        old_text = old_text.replace("\r\n", "\n").replace("\r", "\n")
        new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")
        occurrences = content.count(old_text)
        if occurrences == 1:
            return content.replace(old_text, new_text, 1)
        if occurrences > 1:
            raise ValueError(
                f"{path}: old_text must match exactly once, found {occurrences}"
            )

        matches = self._whitespace_equivalent_spans(content, old_text)
        if len(matches) == 1:
            start, end, matched_text = matches[0]
            replacement = self._align_replacement_indent(
                old_text,
                new_text,
                matched_text,
            )
            self.logger.info(
                "%s: applied unique whitespace-normalized old_text match", path
            )
            return content[:start] + replacement + content[end:]
        if len(matches) > 1:
            raise ValueError(
                f"{path}: old_text has {len(matches)} whitespace-equivalent matches; "
                "use a longer unique repository excerpt"
            )
        raise ValueError(self._no_match_error(path, content, old_text))

    @staticmethod
    def _normalized_lines(text: str) -> tuple[str, ...]:
        lines = text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ()
        dedented = textwrap.dedent("\n".join(lines)).splitlines()
        return tuple(line.rstrip() for line in dedented)

    def _whitespace_equivalent_spans(
        self,
        content: str,
        old_text: str,
    ) -> list[tuple[int, int, str]]:
        target = self._normalized_lines(old_text)
        if not target:
            return []
        content_lines = content.splitlines(keepends=True)
        target_line_count = len(target)
        matches: list[tuple[int, int, str]] = []
        offsets = [0]
        for line in content_lines:
            offsets.append(offsets[-1] + len(line))
        for start_line in range(0, len(content_lines) - target_line_count + 1):
            end_line = start_line + target_line_count
            candidate = "".join(content_lines[start_line:end_line])
            if self._normalized_lines(candidate) == target:
                start = offsets[start_line]
                end = offsets[end_line]
                # A model excerpt normally omits the final newline; keep that
                # newline outside the replaced span unless it explicitly supplied one.
                if not old_text.endswith("\n") and candidate.endswith("\n"):
                    candidate = candidate[:-1]
                    end -= 1
                matches.append((start, end, candidate))
        return matches

    @staticmethod
    def _align_replacement_indent(
        old_text: str,
        new_text: str,
        matched_text: str,
    ) -> str:
        def first_indent(value: str) -> int:
            for line in value.splitlines():
                if line.strip():
                    return len(line) - len(line.lstrip())
            return 0

        delta = first_indent(matched_text) - first_indent(old_text)
        if delta <= 0:
            return new_text
        prefix = " " * delta
        return "\n".join(prefix + line if line.strip() else line for line in new_text.split("\n"))

    @staticmethod
    def _no_match_error(path: str, content: str, old_text: str) -> str:
        requested = next((line.strip() for line in old_text.splitlines() if line.strip()), "")
        lines = content.splitlines()
        if not requested or not lines:
            return f"{path}: old_text must match exactly once, found 0"
        scored = [
            (difflib.SequenceMatcher(None, requested, line.strip()).ratio(), index)
            for index, line in enumerate(lines)
            if line.strip()
        ]
        _, best = max(scored, default=(0.0, 0))
        start = max(0, best - 4)
        end = min(len(lines), best + 5)
        context = "\n".join(
            f"{index + 1}: {lines[index]}" for index in range(start, end)
        )
        return (
            f"{path}: old_text must match exactly once, found 0. "
            f"Closest repository context (copy from this on retry):\n{context}"
        )

    def _validated_edit_path(self, raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe edit path: {raw_path}")
        resolved = (self.env.repo_path / path).resolve()
        try:
            relative = resolved.relative_to(self.env.repo_path)
        except ValueError as exc:
            raise ValueError(f"edit path escapes repository: {raw_path}") from exc
        if not resolved.is_file():
            raise ValueError(f"edit file does not exist: {raw_path}")
        return relative.as_posix()
