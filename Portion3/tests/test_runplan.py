from __future__ import annotations

import pytest

from Portion3 import runplan


def test_matrix_counts_and_uniqueness():
    everything = runplan.all_runs()
    remaining = runplan.new_runs()
    assert len(everything) == 61  # 40 tabular + 21 fine-tune
    assert len(remaining) == 60  # the Portion 2 reference run is excluded
    ids = [run.run_id for run in everything]
    assert len(ids) == len(set(ids))
    assert runplan.EXISTING_FINETUNE_RUNS[0] not in {run.run_id for run in remaining}


def test_tabular_method_coverage():
    runs = runplan.tabular_runs()
    assert sum(run.method == "rules" for run in runs) == 1
    assert sum(run.method == "isolation_forest" for run in runs) == 3
    assert sum(run.method == "lgbm_features" for run in runs) == 15
    assert sum(run.method == "lgbm_embeddings" for run in runs) == 21


def test_every_fraction_and_seed_pair_exists():
    for fraction in runplan.MAIN_FRACTIONS:
        for seed in runplan.SEEDS:
            for method in ("lgbm_features", "lgbm_embeddings", "finetune"):
                matches = [
                    run for run in runplan.all_runs()
                    if run.method == method and run.fraction == fraction and run.seed == seed
                ]
                assert matches, (method, fraction, seed)


def test_ablation_runs_are_five_percent_only():
    for run in runplan.all_runs():
        if run.pretrain_run and run.pretrain_run.startswith("abl_"):
            assert run.fraction == runplan.ABLATION_FRACTION


def test_parse_finetune_folder():
    from Portion3 import portion3_baselines as pb

    assert pb.parse_finetune_folder("ft_main_seed43_20pct_seed43") == ("main_seed43", 0.2, 43)
    assert pb.parse_finetune_folder("ft_abl_no_dt_seed42_5pct_seed44") == ("abl_no_dt_seed42", 0.05, 44)
    with pytest.raises(ValueError):
        pb.parse_finetune_folder("finetune_main")
