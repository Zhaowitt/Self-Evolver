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


@cli.command()
@click.option(
    "--benchmark", "-b", "benchmark_name",
    default="swebench",
    type=click.Choice(["swebench", "swebench_live", "swebench_pro", "multi_swe_bench"]),
    help="Benchmark family to run",
)
@click.option(
    "--dataset", "-d",
    default="lite",
    help="Dataset variant (swebench: lite|verified|full; "
         "swebench_live: lite|full|verified|test; swebench_pro: test; "
         "multi_swe_bench: full|flash)",
)
@click.option("--num-instances", "-n", default=None, type=int,
              help="Instances to run (train stage: number of rollouts). Default: all")
@click.option("--output-dir", "-o", default="./benchmark_results", type=click.Path(),
              help="Run directory for predictions, rollouts, snapshot, and metrics")
@click.option("--workspace-dir", default=None, type=click.Path(),
              help="Directory for cloned benchmark repositories")
@click.option("--split", default="test", help="Dataset split (validated per dataset)")
@click.option("--phase", default="generate",
              type=click.Choice(["generate", "evaluate", "both"]),
              help="Benchmark phase to run")
@click.option("--predictions-path", default=None, type=click.Path(),
              help="Path to the predictions JSON (default: <output-dir>/predictions.json)")
@click.option("--run-id", default="self-evolver", help="Evaluation run id")
@click.option("--model-name", default="self-evolver", help="Prediction model name")
@click.option("--agent-mode", default="mas", type=click.Choice(["single", "mas"]),
              help="single: one-shot baseline; mas: multi-agent repair loop")
@click.option("--skills", default="static", type=click.Choice(["off", "static", "evolve"]),
              help="off: no skills; static: fixed bank; evolve: skill evolution (train)")
@click.option("--memory", default="on", type=click.Choice(["on", "off"]),
              help="Hard-case retrieval and reflection")
@click.option("--task-evolution", default="off", type=click.Choice(["on", "off"]),
              help="TaskPool-driven sampling and focused variants (train stage only)")
@click.option("--controller-mode", default="off", type=click.Choice(["off", "llm"]),
              help="Upstream Controller guidance mode")
@click.option("--stage", default="eval", type=click.Choice(["train", "eval"]),
              help="train enables evolution; eval freezes the skill bank (no writes)")
@click.option("--seed", default=0, type=int, help="RNG seed for task sampling")
@click.option("--test-backend", default="auto",
              type=click.Choice(["auto", "docker", "apptainer", "host"]),
              help="In-loop verification backend (official container semantics)")
@click.option("--hints", is_flag=True, default=False,
              help="Surface human hints_text to the worker (off by default)")
@click.option("--validate-skills", default=0, type=int,
              help="Replay M held-out instances before a reflector skill write")
@click.option("--train-ids", default=None, type=click.Path(exists=True),
              help="File of training ids to exclude from an eval set (contamination guard)")
@click.option("--eval-workers", default=2, help="Workers for the official docker harness")
@click.option("--resume/--no-resume", default=True, help="Resume from existing predictions")
@click.option("--reward-config", default=None, type=click.Path(exists=True),
              help="Reward config file (default: configs/reward_config.yaml)")
@click.option("--cleanup-images/--no-cleanup-images", default=True,
              help="Clean docker env/eval images after each repo batch")
@click.pass_context
def benchmark(
    ctx,
    benchmark_name,
    dataset,
    num_instances,
    output_dir,
    workspace_dir,
    split,
    phase,
    predictions_path,
    run_id,
    model_name,
    agent_mode,
    skills,
    memory,
    task_evolution,
    controller_mode,
    stage,
    seed,
    test_backend,
    hints,
    validate_skills,
    train_ids,
    eval_workers,
    resume,
    reward_config,
    cleanup_images,
):
    """Run a SWE-bench-family benchmark under one experiment configuration."""
    console.print(f"[bold]{benchmark_name} benchmark[/bold]")
    console.print(f"Dataset: {dataset} | Split: {split} | Phase: {phase} | Stage: {stage}")
    console.print(
        f"Experiment: agent={agent_mode} skills={skills} memory={memory} "
        f"task-evolution={task_evolution} controller={controller_mode} "
        f"backend={test_backend} seed={seed}"
    )
    console.print()

    if stage == "eval" and (task_evolution == "on" or skills == "evolve"):
        console.print(
            "[yellow]Note: evolution is train-only; eval runs frozen "
            "(no skill/task writes).[/yellow]"
        )

    config = get_config()
    if phase in {"generate", "both"} and not config.validate_api_key():
        console.print("[red]Error: OPENAI_API_KEY not configured.[/red]")
        console.print("Please set OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    from src.benchmark.swebench_runner import ExperimentConfig
    from src.benchmark.swebench_pro import ProEvaluationUnavailable
    from src.benchmark.swebench_multi import MultiSWEEvaluationUnavailable

    experiment = ExperimentConfig(
        agent_mode=agent_mode,
        skills=skills,
        memory=memory,
        task_evolution=task_evolution,
        controller_mode=controller_mode,
        stage=stage,
        seed=seed,
        test_backend=test_backend,
        hints=hints,
        validate_skills=validate_skills,
        label=f"{benchmark_name}-{dataset}-{stage}",
    )

    if benchmark_name == "swebench":
        from src.benchmark.swebench_runner import create_swebench_runner as make_runner
    elif benchmark_name == "swebench_live":
        from src.benchmark.swebench_live import create_swebench_live_runner as make_runner
    elif benchmark_name == "multi_swe_bench":
        from src.benchmark.swebench_multi import create_multi_swe_bench_runner as make_runner
    else:
        from src.benchmark.swebench_pro import create_swebench_pro_runner as make_runner

    runner = make_runner(
        dataset=dataset,
        output_dir=Path(output_dir),
        workspace_dir=Path(workspace_dir) if workspace_dir else None,
        model_name=model_name,
        run_id=run_id,
        experiment=experiment,
        reward_config=Path(reward_config) if reward_config else None,
        train_ids_path=Path(train_ids) if train_ids else None,
    )

    try:
        result = runner.run_phased_benchmark(
            phase=phase,
            num_instances=num_instances,
            split=split,
            predictions_path=Path(predictions_path) if predictions_path else None,
            run_id=run_id,
            eval_workers=eval_workers,
            resume=resume,
            cleanup_images=cleanup_images,
        )
    except ProEvaluationUnavailable as exc:
        console.print(Panel(str(exc), title="SWE-bench Pro evaluation", border_style="yellow"))
        sys.exit(2)
    except MultiSWEEvaluationUnavailable as exc:
        console.print(Panel(str(exc), title="Multi-SWE-bench evaluation", border_style="yellow"))
        sys.exit(2)

    table = Table(title=f"{benchmark_name} result")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Run directory", str(runner.run_dir))
    table.add_row("Predictions", result["predictions_path"])
    if "predictions" in result:
        predictions = result["predictions"]
        table.add_row("Prediction Total", str(predictions.get("total", 0)))
        table.add_row("Non-empty Patches", str(predictions.get("non_empty", 0)))
        table.add_row("Empty Patches", str(predictions.get("empty", 0)))
    if "evaluation" in result:
        evaluation = result["evaluation"]
        table.add_row("Backend", str(evaluation.get("backend", "")))
        table.add_row("Resolved", str(evaluation.get("resolved_count", 0)))
        table.add_row("Unresolved", str(evaluation.get("unresolved_count", 0)))
        if "infra_error_count" in evaluation:
            table.add_row("Infra Errors", str(evaluation.get("infra_error_count", 0)))
            table.add_row("Patch Errors", str(evaluation.get("patch_error_count", 0)))
        table.add_row("Resolve Rate", f"{evaluation.get('resolve_rate', 0.0):.1%}")
    console.print(table)
    sys.exit(0)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
