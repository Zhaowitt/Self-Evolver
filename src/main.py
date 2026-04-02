"""
Self-Evolver CLI - Main Entry Point.

Provides command-line interface for running the Multi-Agent System.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from src.config import get_config
from src.critic.judge import CriticJudge
from src.environment.models import Issue
from src.environment.project_env import ProjectEnvironment
from src.orchestrator.orchestrator import ExecutionOrchestrator, ExecutionResult

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def print_result(result: ExecutionResult, evaluation=None):
    """Pretty print execution result."""
    status_color = "green" if result.success else "red"
    status_text = "SUCCESS" if result.success else "FAILED"
    
    console.print(Panel(
        f"[bold {status_color}]{status_text}[/bold {status_color}]\n\n"
        f"Issue: {result.issue_id}\n"
        f"Iterations: {result.iterations_used}\n"
        f"Tokens: {result.total_tokens}\n"
        f"Duration: {result.total_duration_ms:.0f}ms",
        title="Execution Result",
    ))
    
    if result.final_patch:
        console.print("\n[bold]Final Patch:[/bold]")
        console.print(f"  Files: {', '.join(result.final_patch.modified_files)}")
        console.print(f"  Changes: +{result.final_patch.added_lines} -{result.final_patch.removed_lines}")
    
    if evaluation:
        console.print(f"\n[bold]Evaluation:[/bold]")
        console.print(f"  {evaluation.summary}")
        if not result.success:
            console.print(f"\n[bold]Reflection:[/bold]")
            console.print(f"  {evaluation.reflection}")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Self-Evolver: Multi-Agent System for Code Repair."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.option("--repo", "-r", required=True, type=click.Path(exists=True), 
              help="Path to the repository")
@click.option("--issue", "-i", required=True, help="Issue description")
@click.option("--issue-id", default="custom-001", help="Issue ID for tracking")
@click.option("--max-iterations", "-m", default=3, help="Maximum iterations")
@click.option("--test-cmd", "-t", default=None, help="Test command to run")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
@click.pass_context
def fix(ctx, repo, issue, issue_id, max_iterations, test_cmd, output):
    """Fix an issue in a local repository."""
    console.print(f"[bold]Self-Evolver Code Repair[/bold]")
    console.print(f"Repository: {repo}")
    console.print(f"Issue: {issue[:100]}...")
    console.print()
    
    # Check API key
    config = get_config()
    if not config.validate_api_key():
        console.print("[red]Error: OPENAI_API_KEY not configured.[/red]")
        console.print("Please set OPENAI_API_KEY environment variable or create .env file.")
        sys.exit(1)
    
    # Initialize environment
    try:
        env = ProjectEnvironment(repo, test_cmd=test_cmd)
    except Exception as e:
        console.print(f"[red]Error initializing environment: {e}[/red]")
        sys.exit(1)
    
    # Create issue
    issue_obj = Issue(
        id=issue_id,
        description=issue,
    )
    
    # Run orchestrator
    console.print("[bold]Starting repair process...[/bold]\n")
    
    orchestrator = ExecutionOrchestrator(env, max_iterations=max_iterations)
    result = orchestrator.run(issue_obj)
    
    # Evaluate
    judge = CriticJudge()
    evaluation = judge.evaluate(result)
    
    # Print results
    print_result(result, evaluation)
    
    # Save output if requested
    if output:
        output_data = {
            "issue_id": result.issue_id,
            "success": result.success,
            "status": result.status.value,
            "iterations_used": result.iterations_used,
            "total_tokens": result.total_tokens,
            "total_duration_ms": result.total_duration_ms,
            "final_patch": result.final_patch.content if result.final_patch else None,
            "evaluation": {
                "success": evaluation.success,
                "failure_type": evaluation.failure_type.value,
                "summary": evaluation.summary,
                "reflection": evaluation.reflection,
            },
        }
        
        Path(output).write_text(json.dumps(output_data, indent=2))
        console.print(f"\nResults saved to: {output}")
    
    sys.exit(0 if result.success else 1)


@cli.command()
@click.option("--repo", "-r", required=True, type=click.Path(exists=True),
              help="Path to the repository")
@click.option("--test-cmd", "-t", default=None, help="Test command")
@click.pass_context
def check(ctx, repo, test_cmd):
    """Check environment setup and run tests."""
    console.print("[bold]Environment Check[/bold]\n")
    
    # Check API key
    config = get_config()
    api_ok = config.validate_api_key()
    console.print(f"API Key: {'[green]OK[/green]' if api_ok else '[red]NOT SET[/red]'}")
    
    # Check repository
    try:
        env = ProjectEnvironment(repo, test_cmd=test_cmd)
        repo_state = env.get_repo_state()
        console.print(f"Repository: [green]OK[/green]")
        console.print(f"  Path: {repo_state.path}")
        console.print(f"  Branch: {repo_state.current_branch}")
        console.print(f"  Commit: {repo_state.current_commit[:8] if repo_state.current_commit else 'N/A'}")
        console.print(f"  Dirty: {repo_state.is_dirty}")
    except Exception as e:
        console.print(f"Repository: [red]ERROR[/red] - {e}")
        sys.exit(1)
    
    # List files
    try:
        py_files = env.list_files("**/*.py")
        console.print(f"  Python files: {len(py_files)}")
    except Exception as e:
        console.print(f"  Python files: [yellow]Could not list ({e})[/yellow]")
    
    # Run tests
    console.print("\n[bold]Running tests...[/bold]")
    test_result = env.run_tests()
    
    status = "[green]PASSED[/green]" if test_result.passed else "[red]FAILED[/red]"
    console.print(f"Test Result: {status}")
    
    if test_result.error_logs:
        console.print("\n[bold]Error Output:[/bold]")
        console.print(test_result.error_logs[:1000])


@cli.command()
@click.pass_context
def config_info(ctx):
    """Show current configuration."""
    config = get_config()
    
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", config.llm.model)
    table.add_row("API Key Set", "Yes" if config.validate_api_key() else "No")
    table.add_row("Max Iterations", str(config.agent.max_iterations))
    table.add_row("Max Tokens", str(config.llm.max_tokens))
    table.add_row("Temperature", str(config.llm.temperature))
    table.add_row("Workspace", str(config.environment.workspace_dir))
    table.add_row("Log Level", config.environment.log_level)
    
    console.print(table)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
