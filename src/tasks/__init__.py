"""Task evolution: families, pool, focused variants, verification (Proposal 2.3)."""

from src.tasks.config import TaskEvolutionConfig, load_task_evolution_config
from src.tasks.families import TASK_FAMILIES, classify_family
from src.tasks.task_pool import TaskPool
from src.tasks.variants import make_focused_variant
from src.tasks.verification import verify_task

__all__ = [
    "TASK_FAMILIES",
    "TaskEvolutionConfig",
    "TaskPool",
    "classify_family",
    "load_task_evolution_config",
    "make_focused_variant",
    "verify_task",
]
