"""
Inspector Worker - Fault Localization Agent.

Responsible for analyzing issues, error logs, and stack traces
to identify the most likely location of bugs.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.controller.injection import format_controller_guidance
from src.environment.models import CodeLocation, ExecutionContext
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient, LLMResponse, Message
from src.workers.base import BaseWorker, WorkerResult

logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Result of the inspection/fault localization process."""
    
    suspected_files: List[str] = field(default_factory=list)
    suspected_locations: List[CodeLocation] = field(default_factory=list)
    root_cause_analysis: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    relevant_code_snippets: dict = field(default_factory=dict)


INSPECTOR_SYSTEM_PROMPT = """You are an expert software engineer specializing in bug localization and root cause analysis.

Your task is to analyze the given issue description, error logs, and code context to:
1. Identify the most likely files and code locations containing the bug
2. Analyze the root cause of the issue
3. Provide specific suggestions for fixing the bug

## Output Format

You MUST respond with a valid JSON object in the following format:
```json
{
    "suspected_files": ["path/to/file1.py", "path/to/file2.py"],
    "suspected_locations": [
        {
            "file_path": "path/to/file.py",
            "start_line": 42,
            "end_line": 50,
            "reason": "This function handles the failing test case"
        }
    ],
    "root_cause_analysis": "Detailed explanation of why the bug occurs...",
    "fix_suggestions": [
        "Suggestion 1: ...",
        "Suggestion 2: ..."
    ],
    "confidence": 0.8
}
```

## Guidelines

1. Focus on the MOST LIKELY locations first (max 3 files, max 5 locations)
2. Be specific about line numbers when possible
3. Explain WHY each location is suspected
4. Consider error messages, stack traces, and test failures carefully
5. Use the available repository tools to inspect real files before finalizing
6. Only include suspected files and locations that are supported by repository evidence
7. If this is a retry attempt, learn from previous failures"""


class Inspector(BaseWorker):
    """Inspector worker for fault localization."""
    
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(llm_client=llm_client, name="Inspector")
        self.env = env
        agent_config = get_config().agent
        self.max_tool_calls = max(0, agent_config.inspector_max_tool_calls)
        self.read_max_lines = max(1, agent_config.inspector_read_max_lines)
        self.read_max_chars = max(1000, agent_config.inspector_read_max_chars)
    
    @property
    def system_prompt(self) -> str:
        return INSPECTOR_SYSTEM_PROMPT
    
    def execute(self, context: ExecutionContext) -> WorkerResult[InspectionResult]:
        """Execute fault localization."""
        self.logger.info(f"Starting inspection for issue: {context.issue.id}")
        
        try:
            user_message = self._build_analysis_prompt(context)
            additional_context = None
            if context.has_previous_attempt:
                additional_context = self._build_retry_context(context)
            
            response, tool_trace = self._call_llm_with_repo_tools(
                user_message,
                additional_context,
            )
            result = self._parse_response(response.content)
            result = self._validate_result_paths(result)
            result.relevant_code_snippets = self._fetch_code_snippets(result)
            
            self.logger.info(
                f"Inspection complete: {len(result.suspected_files)} files, "
                f"{len(result.suspected_locations)} locations, "
                f"{len(tool_trace)} tool calls"
            )
            
            return WorkerResult(
                success=True,
                data=result,
                llm_response=response,
                metadata={"tool_trace": tool_trace},
            )
            
        except Exception as e:
            self.logger.error(f"Inspection failed: {e}")
            return WorkerResult(success=False, error=str(e))
    
    def _build_analysis_prompt(self, context: ExecutionContext) -> str:
        """Build the main analysis prompt."""
        parts = []
        parts.append("## Issue Description")
        parts.append(context.issue.description)

        controller_guidance = format_controller_guidance(
            context.metadata.get("controller_signal")
        )
        if controller_guidance:
            parts.append("\n" + controller_guidance)
        
        if context.issue.hints:
            parts.append("\n## Hints")
            parts.append(context.issue.hints)
        
        parts.append(f"\n## Repository: {context.repo_state.path.name}")
        
        try:
            py_files = self.env.list_files("**/*.py")[:50]
            parts.append("\n## Python Files in Repository:")
            parts.append("\n".join(f"- {f}" for f in py_files))
        except Exception as e:
            self.logger.warning(f"Could not list files: {e}")
        
        if context.last_test_result and context.last_test_result.error_logs:
            parts.append("\n## Error Logs:")
            parts.append(f"```\n{context.last_test_result.error_logs[:3000]}\n```")
        
        return "\n".join(parts)
    
    def _build_retry_context(self, context: ExecutionContext) -> str:
        """Build context for retry attempts."""
        parts = [f"## This is attempt #{context.iteration + 1}"]
        
        error_context = self._format_error_context(context)
        if error_context:
            parts.append(error_context)
        
        test_context = self._format_test_results(context)
        if test_context:
            parts.append(test_context)
        
        if context.previous_patches:
            parts.append("\n## Previous Patches (failed):")
            for i, patch in enumerate(context.previous_patches, 1):
                parts.append(f"\n### Patch {i}:")
                parts.append(f"Modified files: {', '.join(patch.modified_files)}")
                parts.append(f"```diff\n{patch.content[:1500]}\n```")
        
        parts.append("\n**Please analyze why previous attempts failed and try a different approach.**")
        return "\n".join(parts)

    def _call_llm_with_repo_tools(
        self,
        user_message: str,
        additional_context: Optional[str] = None,
    ) -> tuple[LLMResponse, List[Dict[str, Any]]]:
        """Call the LLM with bounded read-only repository tools."""
        if additional_context:
            user_message = f"{user_message}\n\n{additional_context}"

        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]
        tool_trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tools = self._repo_tool_schemas()
        tool_calls_used = 0
        forced_tool_prompt_sent = False

        for _ in range(self.max_tool_calls + 1):
            response = self.llm_client.chat(
                messages,
                tools=tools,
                tool_choice="auto",
            )
            self._accumulate_usage(total_usage, response)

            if not response.tool_calls:
                if (
                    not tool_trace
                    and self.max_tool_calls > 0
                    and not forced_tool_prompt_sent
                ):
                    forced_tool_prompt_sent = True
                    messages.append(Message(role="assistant", content=response.content))
                    messages.append(
                        Message(
                            role="user",
                            content=(
                                "Before finalizing, inspect the repository with "
                                "grep_repo, list_dir, or read_file. At minimum, "
                                "read the most likely source file."
                            ),
                        )
                    )
                    continue
                response.usage = total_usage
                return response, tool_trace

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                if tool_calls_used >= self.max_tool_calls:
                    result = self._tool_error("Tool call budget exhausted.")
                else:
                    result = self._execute_repo_tool(tool_call)
                    tool_calls_used += 1

                tool_trace.append({
                    "name": self._tool_call_name(tool_call),
                    "arguments": self._tool_call_arguments_text(tool_call),
                    "result_preview": result[:500],
                })
                messages.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

            if tool_calls_used >= self.max_tool_calls:
                break

        messages.append(
            Message(
                role="user",
                content=(
                    "Repository tool budget is exhausted. Respond now with the "
                    "required final JSON object using only the evidence already read."
                ),
            )
        )
        final_response = self.llm_client.chat(messages)
        self._accumulate_usage(total_usage, final_response)
        final_response.usage = total_usage
        return final_response, tool_trace

    @staticmethod
    def _accumulate_usage(total_usage: Dict[str, int], response: LLMResponse) -> None:
        """Accumulate token usage across a tool loop."""
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[key] += response.usage.get(key, 0)

    def _repo_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible schemas for read-only repository tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a repository file by repo-relative path. Returns "
                        "line-numbered text. Use this before finalizing locations."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep_repo",
                    "description": (
                        "Search repository text for a literal or regex pattern. "
                        "Returns repo-relative file, line, and matching text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string", "default": "."},
                            "glob": {"type": "string"},
                            "max_matches": {"type": "integer", "default": 50},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": (
                        "List files and directories under a repo-relative directory."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "default": "."},
                        },
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _execute_repo_tool(self, tool_call: Dict[str, Any]) -> str:
        """Dispatch a repository tool call and return a text result."""
        name = self._tool_call_name(tool_call)
        arguments_text = self._tool_call_arguments_text(tool_call)
        try:
            arguments = json.loads(arguments_text or "{}")
        except json.JSONDecodeError as e:
            return self._tool_error(f"Invalid JSON arguments: {e}")

        try:
            if name == "read_file":
                return self._tool_read_file(
                    path=arguments.get("path", ""),
                    start_line=arguments.get("start_line"),
                    end_line=arguments.get("end_line"),
                )
            if name == "grep_repo":
                return self._tool_grep_repo(
                    pattern=arguments.get("pattern", ""),
                    path=arguments.get("path", "."),
                    glob=arguments.get("glob"),
                    max_matches=arguments.get("max_matches", 50),
                )
            if name == "list_dir":
                return self._tool_list_dir(path=arguments.get("path", "."))
            return self._tool_error(f"Unknown tool: {name}")
        except Exception as e:
            self.logger.warning(f"Inspector tool {name} failed: {e}")
            return self._tool_error(str(e))

    @staticmethod
    def _tool_call_name(tool_call: Dict[str, Any]) -> str:
        return str(tool_call.get("function", {}).get("name", ""))

    @staticmethod
    def _tool_call_arguments_text(tool_call: Dict[str, Any]) -> str:
        return str(tool_call.get("function", {}).get("arguments", "{}"))

    @staticmethod
    def _tool_error(message: str) -> str:
        return f"ERROR: {message}"

    def _resolve_repo_path(self, path: str) -> Path:
        """Resolve and validate a repo-relative path."""
        if not path or not isinstance(path, str):
            raise ValueError("path is required")
        normalized = path.replace("\\", "/").strip()
        if normalized in {"", "."}:
            normalized = "."
        if normalized.startswith("/") or normalized.startswith("~"):
            raise ValueError("absolute paths are not allowed")

        parts = [part for part in normalized.split("/") if part not in {"", "."}]
        if any(part == ".." for part in parts):
            raise ValueError("parent directory traversal is not allowed")
        if parts and parts[0] == ".git":
            raise ValueError("reading .git is not allowed")

        full_path = (self.env.repo_path / normalized).resolve()
        try:
            full_path.relative_to(self.env.repo_path)
        except ValueError as e:
            raise ValueError("path escapes repository") from e
        return full_path

    def _repo_relative(self, path: Path) -> str:
        return str(path.relative_to(self.env.repo_path)).replace("\\", "/")

    def _tool_read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read a repo file and return line-numbered text."""
        file_path = self._resolve_repo_path(path)
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if file_path.stat().st_size > max(self.read_max_chars * 10, 2_000_000):
            raise ValueError("file is too large to read")

        content = file_path.read_text(encoding="utf-8")
        if "\x00" in content:
            raise ValueError("binary files are not supported")

        lines = content.splitlines()
        start = int(start_line) if start_line else 1
        if start < 1:
            raise ValueError("start_line must be >= 1")
        end = int(end_line) if end_line else min(len(lines), start + self.read_max_lines - 1)
        if end < start:
            raise ValueError("end_line must be >= start_line")
        end = min(end, start + self.read_max_lines - 1)

        selected = lines[start - 1:end]
        rendered = "\n".join(
            f"{line_no:>6}: {line}"
            for line_no, line in enumerate(selected, start=start)
        )
        if len(rendered) > self.read_max_chars:
            rendered = rendered[:self.read_max_chars] + "\n... (truncated)"

        header = f"FILE: {self._repo_relative(file_path)} lines {start}-{end}"
        return f"{header}\n{rendered}"

    def _tool_list_dir(self, path: str = ".") -> str:
        """List a repo directory."""
        dir_path = self._resolve_repo_path(path or ".")
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        entries = []
        for child in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if child.name == ".git":
                continue
            kind = "dir" if child.is_dir() else "file"
            entries.append(f"{kind}\t{self._repo_relative(child)}")
            if len(entries) >= 200:
                entries.append("... (truncated)")
                break
        return "\n".join(entries) if entries else "(empty directory)"

    def _tool_grep_repo(
        self,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        max_matches: int = 50,
    ) -> str:
        """Search repository text with ripgrep and a Python fallback."""
        if not pattern or not isinstance(pattern, str):
            raise ValueError("pattern is required")
        search_path = self._resolve_repo_path(path or ".")
        max_matches = max(1, min(int(max_matches or 50), 200))

        command = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(max_matches),
        ]
        if glob:
            command.extend(["--glob", str(glob)])
        command.extend([pattern, str(search_path)])

        try:
            result = subprocess.run(
                command,
                cwd=self.env.repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode in {0, 1}:
                output = result.stdout.strip()
                if not output:
                    return "No matches."
                return self._format_rg_output(output, max_matches)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return self._grep_repo_fallback(pattern, search_path, glob, max_matches)

    def _format_rg_output(self, output: str, max_matches: int) -> str:
        """Normalize ripgrep output to repo-relative paths."""
        formatted = []
        for line in output.splitlines()[:max_matches]:
            parts = line.rsplit(":", 2)
            if len(parts) == 3:
                file_part, line_no, rest = parts
                try:
                    rel = self._repo_relative(Path(file_part).resolve())
                    formatted.append(f"{rel}:{line_no}:{rest}")
                    continue
                except ValueError:
                    pass
            formatted.append(line)
        if len(output.splitlines()) > max_matches:
            formatted.append("... (truncated)")
        return "\n".join(formatted)

    def _grep_repo_fallback(
        self,
        pattern: str,
        search_path: Path,
        glob: Optional[str],
        max_matches: int,
    ) -> str:
        """Fallback grep implementation when ripgrep is unavailable."""
        matches = []
        regex = re.compile(pattern)
        paths = [search_path] if search_path.is_file() else search_path.rglob("*")
        for candidate in paths:
            if len(matches) >= max_matches:
                break
            if not candidate.is_file() or ".git" in candidate.parts:
                continue
            rel = self._repo_relative(candidate)
            if glob and not candidate.match(glob):
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{line_no}:{line[:300]}")
                    if len(matches) >= max_matches:
                        break
        return "\n".join(matches) if matches else "No matches."

    def _validate_result_paths(self, result: InspectionResult) -> InspectionResult:
        """Drop hallucinated suspected paths before handing context to generator."""
        valid_files = []
        for file_path in result.suspected_files:
            if self._is_existing_repo_file(file_path) and file_path not in valid_files:
                valid_files.append(file_path)

        valid_locations = []
        for location in result.suspected_locations:
            if self._is_existing_repo_file(location.file_path):
                valid_locations.append(location)
                if location.file_path not in valid_files:
                    valid_files.append(location.file_path)

        result.suspected_files = valid_files[:3]
        result.suspected_locations = valid_locations[:5]
        return result

    def _is_existing_repo_file(self, file_path: str) -> bool:
        try:
            resolved = self._resolve_repo_path(file_path)
        except ValueError:
            return False
        return resolved.exists() and resolved.is_file()
    
    def _parse_response(self, content: str) -> InspectionResult:
        """Parse the LLM response into InspectionResult."""
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return self._parse_fallback(content)
        
        try:
            data = json.loads(json_str)
            locations = []
            for loc in data.get("suspected_locations", []):
                locations.append(CodeLocation(
                    file_path=loc.get("file_path", ""),
                    start_line=loc.get("start_line", 1),
                    end_line=loc.get("end_line"),
                    snippet=loc.get("reason", ""),
                ))
            
            return InspectionResult(
                suspected_files=data.get("suspected_files", []),
                suspected_locations=locations,
                root_cause_analysis=data.get("root_cause_analysis", ""),
                fix_suggestions=data.get("fix_suggestions", []),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse error: {e}")
            return self._parse_fallback(content)
    
    def _parse_fallback(self, content: str) -> InspectionResult:
        """Fallback parsing when JSON extraction fails."""
        file_pattern = r'[\w/]+\.py'
        files = list(set(re.findall(file_pattern, content)))
        return InspectionResult(
            suspected_files=files[:5],
            root_cause_analysis=content[:1000],
            fix_suggestions=["Review the analysis above for fix suggestions"],
            confidence=0.3,
        )
    
    def _fetch_code_snippets(self, result: InspectionResult) -> dict:
        """Fetch code snippets for suspected locations."""
        snippets = {}
        for location in result.suspected_locations[:5]:
            try:
                start = max(1, location.start_line - 5)
                end = (location.end_line or location.start_line) + 5
                content = self.env.get_file_content_with_lines(
                    location.file_path, start_line=start, end_line=end
                )
                key = f"{location.file_path}:{start}-{end}"
                snippets[key] = content
            except FileNotFoundError:
                self.logger.warning(f"File not found: {location.file_path}")
            except Exception as e:
                self.logger.warning(f"Error fetching snippet: {e}")
        return snippets
