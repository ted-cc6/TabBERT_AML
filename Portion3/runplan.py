"""Single source of truth for the Portion 3 run matrix.

Both the command line driver and the documentation generator read the plan from
here, so the runbook cannot drift from the code. Import this module instead of
typing run names by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

MAIN_CHECKPOINTS = ("main_seed42", "main_seed43", "main_seed44")
MAIN_FRACTIONS = (0.01, 0.02, 0.05, 0.2, 1.0)
SEEDS = (42, 43, 44)
ABLATION_CHECKPOINTS = ("abl_whole_mask_seed42", "abl_no_dt_seed42")
ABLATION_FRACTION = 0.05

# The reference fine-tuning run Portion 2 produced (HANDOFF.md, section 1).
EXISTING_FINETUNE_RUNS = ("ft_main_seed42_5pct_seed42",)


def pct(fraction: float) -> str:
    """Fraction as it appears in run ids and folder names: 0.05 -> '5pct'."""
    return f"{fraction * 100:g}pct"


def pretrain_run_for_seed(seed: int) -> str:
    """Downstream seed s uses the main checkpoint pretrained with seed s."""
    return f"main_seed{seed}"


@dataclass(frozen=True)
class Run:
    run_id: str
    stage: str  # "tabular" or "finetune"
    method: str  # rules | isolation_forest | lgbm_features | lgbm_embeddings | finetune
    fraction: float | None
    seed: int | None
    pretrain_run: str | None

    @property
    def is_label_free(self) -> bool:
        return self.method in ("rules", "isolation_forest")


def tabular_runs() -> list[Run]:
    runs: list[Run] = [
        # Rules have no random component: one row in the log, evaluated on valid/test.
        Run("rules", "tabular", "rules", None, None, None),
    ]
    # Isolation Forest is unsupervised as well, but the forest itself is seeded.
    for seed in SEEDS:
        runs.append(Run(f"if_seed{seed}", "tabular", "isolation_forest", None, seed, None))
    for fraction in MAIN_FRACTIONS:
        for seed in SEEDS:
            runs.append(Run(
                f"lgbm_features_{pct(fraction)}_seed{seed}",
                "tabular", "lgbm_features", fraction, seed, None,
            ))
            runs.append(Run(
                f"lgbm_embeddings_{pct(fraction)}_seed{seed}",
                "tabular", "lgbm_embeddings", fraction, seed, pretrain_run_for_seed(seed),
            ))
    # Experiment C through the Experiment A pipeline: ablation embeddings at 5% labels.
    for checkpoint in ABLATION_CHECKPOINTS:
        for seed in SEEDS:
            runs.append(Run(
                f"lgbm_embeddings_{checkpoint}_{pct(ABLATION_FRACTION)}_seed{seed}",
                "tabular", "lgbm_embeddings", ABLATION_FRACTION, seed, checkpoint,
            ))
    return runs


def finetune_runs() -> list[Run]:
    runs: list[Run] = []
    for fraction in MAIN_FRACTIONS:
        for seed in SEEDS:
            checkpoint = pretrain_run_for_seed(seed)
            runs.append(Run(
                f"ft_{checkpoint}_{pct(fraction)}_seed{seed}",
                "finetune", "finetune", fraction, seed, checkpoint,
            ))
    for checkpoint in ABLATION_CHECKPOINTS:
        for seed in SEEDS:
            runs.append(Run(
                f"ft_{checkpoint}_{pct(ABLATION_FRACTION)}_seed{seed}",
                "finetune", "finetune", ABLATION_FRACTION, seed, checkpoint,
            ))
    return runs


def all_runs() -> list[Run]:
    return tabular_runs() + finetune_runs()


def new_runs() -> list[Run]:
    """Every run still to produce, excluding the reference run Portion 2 finished."""
    return [run for run in all_runs() if run.run_id not in EXISTING_FINETUNE_RUNS]


def plan_table() -> list[dict]:
    """Rows for RUN_PLAN.md and the `plan` command."""
    return [
        {
            "run_id": run.run_id,
            "stage": run.stage,
            "method": run.method,
            "fraction": "" if run.fraction is None else pct(run.fraction),
            "seed": "" if run.seed is None else run.seed,
            "pretrain_run": run.pretrain_run or "",
        }
        for run in all_runs()
    ]


def markdown_table() -> str:
    header = "| # | run_id | stage | method | fraction | seed | pretrain run |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- |\n"
    lines = []
    for i, row in enumerate(plan_table(), start=1):
        lines.append(
            f"| {i} | `{row['run_id']}` | {row['stage']} | {row['method']} | "
            f"{row['fraction'] or '-'} | {row['seed'] or '-'} | {row['pretrain_run'] or '-'} |"
        )
    return header + "\n".join(lines) + "\n"
