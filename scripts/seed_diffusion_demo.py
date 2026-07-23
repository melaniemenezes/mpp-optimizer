"""Populate the diffusion-study campaign with synthetic MPT data (no lab data needed).

Matches the professor's spec: 5-lipid molar ratios + liposome size + zeta as inputs, and the
mucin-diffusion metrics (D, D1, Dalpha, alpha, net-to-path) as outputs.

Run from the project root:  python scripts/seed_diffusion_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mpp import service
from mpp.benchmark import make_diffusion_study_campaign, simulate_diffusion
from mpp.db import init_db
from mpp.optimizer import build_optimizer
from mpp.schema import ExperimentRecord


def main(n: int = 30) -> None:
    init_db(seed=True)
    cfg = make_diffusion_study_campaign()
    cid = service.create_campaign(cfg)
    opt = build_optimizer(cfg, rng_seed=7)
    rng = np.random.default_rng(99)

    # Space-filling over inputs (lipid fractions + size + zeta), then simulate the assay.
    batch = opt.suggest_batch([], n=n)
    for i, b in enumerate(batch):
        ro = simulate_diffusion(b["composition"], b["process"], rng=rng)
        rec = ExperimentRecord(composition=b["composition"], process=b["process"], readouts=ro,
                               label="synthetic MPT", plate="diffusion", well=f"F{i+1}")
        service.add_experiment(cid, rec, source="demo", status="completed")

    records = service.records_for_optimizer(cid)
    print(f"Seeded diffusion-study campaign #{cid}: {n} formulations.")
    print("Readouts available for characterisation:", opt.available_readouts(records))
    sens = opt.sensitivities(records, readout="alpha_exponent")
    top = sorted(sens["importances"].items(), key=lambda kv: -kv[1])[:3]
    print("Top drivers of alpha (mobility regime):", [(k, round(v, 2)) for k, v in top])
    print("\nLaunch the app with:  streamlit run app.py  → Model & Insights → pick a readout.")


if __name__ == "__main__":
    main()
