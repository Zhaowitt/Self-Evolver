"""
Patch Generator Worker - Code Patch Generation Agent.

Responsible for generating code patches based on inspection results.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.environment.models import ExecutionContext, PatchInfo
from src.environment.project_env import ProjectEnvironment
from src.llm.client import LLMClient
from src.workers.base import BaseWorker, WorkerResult
from src.workers.inspector import InspectionResult

logger = logging.getLogger(__name__)


@dataclass
class PatchResult:
    """Result of patch generation."""
    
    patch_content: str = ""
    modified_files: List[str] = field(default_factory=list)
    explanation: str = ""
    patch_info: Optional[PatchInfo] = None


PATCH_GENERATOR_SYSTEM_PROMPT = """You are an expert software engineer specializing in bug fixing and code patching.

Your task is to generate a minimal, correct patch to fix the identified bug.

## Output Format

You MUST respond with a valid JSON object containing the patch in unified diff format:
```json
{
    "explanation": "Brief explanation of what the patch does and why",
    "modified_files": ["path/to/file.py"],
    "patch": "--- a/path/to/file.py\\n+++ b/path/to/file.py\\n@@ -line,count +line,count @@\\n context line\\n-removed line\\n+added line\\n context line"
}
```

## CRITICAL: Unified Diff Line Prefix Rules

Every line in a hunk body MUST start with EXACTLY ONE prefix character:
- ` ` (SPACE) — unchanged context line (this space is the prefix, NOT part of the code)
- `-` — removed line
- `+` — added line

**WRONG** (missing space prefix on context lines):
```
@@ -10,4 +10,4 @@
def example_function(x):
    # Some context
-    return x + 1
+    return x * 2
    # More context
```

**CORRECT** (every context line starts with a space):
```
@@ -10,6 +10,6 @@
 def example_function(x):
     # Some context
-    return x + 1  # Bug: should be x * 2
+    return x * 2  # Fixed
     # More context
 
```

Key rules:
1. Even a blank/empty line in the context MUST be written as a single space `' '`, never as an empty string `''`
2. The `@@ -old_start,old_count +new_start,new_count @@` counts MUST be accurate:
   - `old_count` = number of context lines + number of removed lines
   - `new_count` = number of context lines + number of added lines
3. Include 3 context lines before and after each change

## Guidelines

1. Generate patches in UNIFIED DIFF format (git diff style)
2. Make MINIMAL changes - only fix what's necessary
3. Preserve existing code style and formatting
4. Include sufficient context lines (3 lines before and after each change)
5. If multiple files need changes, include all diffs in one patch string
6. Test your logic mentally before generating the patch
7. If this is a retry, avoid the same mistakes from previous attempts"""


class PatchGenerator(BaseWorker):
    """Patch Generator worker for creating code fixes."""
    
    def __init__(
        self,
        env: ProjectEnvironment,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(llm_client=llm_client, name="PatchGenerator")
        self.env = env
    
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
            
            response = self._call_llm(user_message, additional_context)
            result = self._parse_response(response.content)
            
            if result.patch_content:
                result.patch_info = PatchInfo.from_diff(result.patch_content)
            
            self.logger.info(
                f"Patch generated: {len(result.modified_files)} files modified"
            )
            
            return WorkerResult(success=True, data=result, llm_response=response)
            
        except Exception as e:
            self.logger.error(f"Patch generation failed: {e}")
            return WorkerResult(success=False, error=str(e))
    
    def _build_patch_prompt(
        self,
        context: ExecutionContext,
        inspection_result: Optional[InspectionResult],
    ) -> str:
        """Build the patch generation prompt."""
        parts = []
        
        parts.append("## Issue Description")
        parts.append(context.issue.description)
        
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
                    # Limit content size
                    if len(content) > 5000:
                        content = content[:5000] + "\n... (truncated)"
                    parts.append(f"\n### {file_path}")
                    parts.append(f"```python\n{content}\n```")
                except FileNotFoundError:
                    self.logger.warning(f"File not found: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
        
        parts.append("\n## Task")
        parts.append("Generate a minimal patch in unified diff format to fix this issue.")
        
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
                    "Focus on producing a syntactically valid unified diff with correct "
                    "file headers, hunk headers, context prefixes, and complete hunks."
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
        
        if context.last_test_result:
            parts.append("\n## Last Test Result:")
            if context.last_test_result.error_logs:
                parts.append(f"```\n{context.last_test_result.error_logs[:1500]}\n```")
        
        parts.append("\n**Analyze why previous patches failed and generate a corrected version.**")
        return "\n".join(parts)
    
    def _parse_response(self, content: str) -> PatchResult:
        """Parse LLM response into PatchResult."""
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
                patch = data.get("patch", "")
                # Unescape newlines if needed
                patch = patch.replace("\\n", "\n")
                
                return PatchResult(
                    patch_content=patch,
                    modified_files=data.get("modified_files", []),
                    explanation=data.get("explanation", ""),
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract diff directly
        diff_match = re.search(r'```(?:diff)?\s*(---.*?)\s*```', content, re.DOTALL)
        if diff_match:
            patch = diff_match.group(1)
            patch_info = PatchInfo.from_diff(patch)
            return PatchResult(
                patch_content=patch,
                modified_files=patch_info.modified_files,
                explanation="Extracted from response",
            )
        
        # Last resort: look for unified diff pattern anywhere
        diff_pattern = r'(--- a/.*?\n\+\+\+ b/.*?\n@@.*?(?:\n[-+ ].*)*)'
        diff_match = re.search(diff_pattern, content, re.DOTALL)
        if diff_match:
            patch = diff_match.group(1)
            patch_info = PatchInfo.from_diff(patch)
            return PatchResult(
                patch_content=patch,
                modified_files=patch_info.modified_files,
                explanation="Extracted from response",
            )
        
        self.logger.warning("Could not parse patch from response")
        return PatchResult(
            patch_content="",
            explanation=f"Failed to parse patch. Raw response: {content[:500]}",
        )
