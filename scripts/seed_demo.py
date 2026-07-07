"""Populate a synthetic demo campaign so the whole loop works with zero lab data.

Run from the project root:  python scripts/seed_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mpp import service, storage
from mpp.benchmark import make_demo_campaign, simulate
from mpp.db import init_db
from mpp.optimizer import build_optimizer
from mpp.schema import ExperimentRecord


def main(n_seed: int = 24, n_bo: int = 12) -> None:
    init_db(seed=True)
    cfg = make_demo_campaign()
    cid = service.create_campaign(cfg)
    opt = build_optimizer(cfg, rng_seed=123)
    rng = np.random.default_rng(2024)

    # Round 0: space-filling design, simulate the assay, store as completed.
    batch0 = opt.suggest_batch([], n=n_seed)
    first_id = None
    for i, b in enumerate(batch0):
        ro = simulate(b["composition"], b["process"], rng=rng)
        rec = ExperimentRecord(composition=b["composition"], process=b["process"], readouts=ro,
                               label="seed (space-filling)", plate="plate0", well=f"S{i+1}")
        eid = service.add_experiment(cid, rec, source="demo", status="completed")
        first_id = first_id or eid

    # Attach a small note to the first experiment to demonstrate file storage.
    if first_id:
        note = b"# DLS note\nZ-average and PDI measured on Malvern ZS.\nReplicate n=3.\n"
        meta = storage.save_bytes(first_id, "dls_note.md", note, "text/markdown")
        service.add_attachment(first_id, meta)

    # Round 1: Bayesian-optimized batch on top of the seed data.
    records = service.records_for_optimizer(cid)
    batch1 = opt.suggest_batch(records, n=n_bo)
    for i, b in enumerate(batch1):
        ro = simulate(b["composition"], b["process"], rng=rng)
        rec = ExperimentRecord(composition=b["composition"], process=b["process"], readouts=ro,
                               label="BO round 1", plate="plate1", well=f"B{i+1}")
        service.add_experiment(cid, rec, source="demo", status="completed")

    records = service.records_for_optimizer(cid)
    best = opt.recommend(records, k=1)
    print(f"Seeded demo campaign #{cid}: {n_seed + n_bo} completed experiments.")
    if best:
        b = best[0]
        print("Best so far:", {k: round(v, 3) for k, v in b["readouts"].items() if k in
                                [o.readout for o in cfg.objectives]})
        print("Composition:", b["composition"])
    print("\nLaunch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
