"""
Inspector Worker - Fault Localization Agent.

Responsible for analyzing issues, error logs, and stack traces
to identify the most likely location of bugs.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.environment.models import CodeLocation, ExecutionContext
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
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
5. If this is a retry attempt, learn from previous failures"""


class Inspector(BaseWorker):
    """Inspector worker for fault localization."""
    
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(llm_client=llm_client, name="Inspector")
        self.env = env
    
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
            
            response = self._call_llm(user_message, additional_context)
            result = self._parse_response(response.content)
            result.relevant_code_snippets = self._fetch_code_snippets(result)
            
            self.logger.info(
                f"Inspection complete: {len(result.suspected_files)} files, "
                f"{len(result.suspected_locations)} locations"
            )
            
            return WorkerResult(success=True, data=result, llm_response=response)
            
        except Exception as e:
            self.logger.error(f"Inspection failed: {e}")
            return WorkerResult(success=False, error=str(e))
    
    def _build_analysis_prompt(self, context: ExecutionContext) -> str:
        """Build the main analysis prompt."""
        parts = []
        parts.append("## Issue Description")
        parts.append(context.issue.description)
        
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
