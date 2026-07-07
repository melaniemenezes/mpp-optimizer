"""Optimizer behaviour tests against the synthetic benchmark."""
import numpy as np

from mpp.benchmark import make_demo_campaign, simulate
from mpp.optimizer import build_optimizer
from mpp.schema import (
    CampaignConfig,
    ComponentSpec,
    ConstraintSpec,
    ObjectiveSpec,
    ProcessParamSpec,
)


def _simulate_batch(batch, rng):
    return [
        {"composition": b["composition"], "process": b["process"],
         "readouts": simulate(b["composition"], b["process"], rng=rng)}
        for b in batch
    ]


def test_cold_start_is_space_filling_and_valid():
    cfg = make_demo_campaign()
    opt = build_optimizer(cfg, rng_seed=1)
    batch = opt.suggest_batch([], n=8)
    assert all(b["_method"] == "space-filling" for b in batch)
    for b in batch:
        assert abs(sum(b["composition"].values()) - 1.0) < 1e-9
        # free fractions respect their bounds
        assert 0.10 - 1e-6 <= b["composition"]["Cholesterol"] <= 0.50 + 1e-6
        assert 1.0 - 1e-6 <= b["process"]["total_lipid_conc"] <= 20.0 + 1e-6


def test_switches_to_bayesopt_with_enough_data():
    cfg = make_demo_campaign()
    opt = build_optimizer(cfg, rng_seed=2)
    rng = np.random.default_rng(0)
    records = _simulate_batch(opt.suggest_batch([], n=12), rng)
    batch = opt.suggest_batch(records, n=4, n_candidates=800)
    assert all(b["_method"] == "bayesopt" for b in batch)
    for b in batch:
        assert abs(sum(b["composition"].values()) - 1.0) < 1e-9


def test_optimization_finds_good_compositions():
    cfg = make_demo_campaign()
    opt = build_optimizer(cfg, rng_seed=3)
    rng = np.random.default_rng(1)
    records = _simulate_batch(opt.suggest_batch([], n=12), rng)
    for _ in range(3):  # a few BO rounds
        records += _simulate_batch(opt.suggest_batch(records, n=6, n_candidates=1200), rng)
    best = max(r["readouts"]["mucus_penetration"] for r in records)
    # the constrained optimum is high; BO should comfortably clear this bar
    assert best > 45.0


def test_constraints_and_pareto():
    cfg = make_demo_campaign()
    opt = build_optimizer(cfg, rng_seed=4)
    rng = np.random.default_rng(2)
    records = _simulate_batch(opt.suggest_batch([], n=10), rng)
    par = opt.pareto(records)
    assert len(par["is_pareto"]) == len([r for r in records if opt.is_complete(r)])
    # a record that violates the zeta constraint must be flagged infeasible
    bad = {"composition": {"DSPC": 0.4, "DOTAP": 0.3, "Cholesterol": 0.2, "DSPE-PEG2000": 0.1},
           "process": {"total_lipid_conc": 10.0, "flow_ratio": 3.0},
           "readouts": {"mucus_penetration": 5.0, "encapsulation_pct": 50.0, "size_nm": 120.0,
                        "pdi": 0.2, "zeta_mv": 40.0}}
    feas = opt._feasible_mask([bad])
    assert feas[0] == False  # zeta 40 mV is outside [-10, 10]


def test_sensitivities_shape():
    cfg = make_demo_campaign()
    opt = build_optimizer(cfg, rng_seed=5)
    rng = np.random.default_rng(3)
    records = _simulate_batch(opt.suggest_batch([], n=14), rng)
    sens = opt.sensitivities(records)
    assert set(sens["importances"].keys()) == set(cfg.dim_names())
    assert abs(sum(sens["importances"].values()) - 1.0) < 1e-6
    assert 1 <= len(sens["partial_dependence"]) <= 4


def test_pure_mixture_no_filler_normalizes():
    cfg = CampaignConfig(
        name="mix",
        components=[ComponentSpec(lipid="DSPC", low=0.2, high=0.8),
                    ComponentSpec(lipid="DOPC", low=0.2, high=0.8)],
        process_params=[ProcessParamSpec(name="flow_ratio", low=1.0, high=5.0)],
        objectives=[ObjectiveSpec(readout="size_nm", direction="target", target=100.0)],
    )
    opt = build_optimizer(cfg, rng_seed=6)
    batch = opt.suggest_batch([], n=5)
    for b in batch:
        assert abs(sum(b["composition"].values()) - 1.0) < 1e-6
