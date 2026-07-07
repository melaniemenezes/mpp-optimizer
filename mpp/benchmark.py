"""Synthetic mucus-penetration ground truth for demos and tests.

This stands in for the wet-lab + 96-well assay when no real data exists yet. It maps a
formulation (composition + process params) to plausible readouts with realistic trade-offs:
  * PEG near a sweet spot and near-neutral zeta maximise mucus penetration (muco-inert),
  * cholesterol boosts encapsulation/retention but too much PEG hurts them,
  * size is driven by lipid concentration and flow ratio.
The optimizer has to navigate the tension between penetration and cargo retention.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .schema import (
    CampaignConfig,
    ComponentSpec,
    ConstraintSpec,
    ObjectiveSpec,
    ProcessParamSpec,
)


def _g(x: float, mu: float, sd: float) -> float:
    return math.exp(-((x - mu) / sd) ** 2)


def simulate(composition: Dict[str, float], process: Dict[str, float],
             rng: Optional[np.random.Generator] = None, noise: bool = True) -> Dict[str, float]:
    f_peg = float(composition.get("DSPE-PEG2000", 0.0))
    f_chol = float(composition.get("Cholesterol", 0.0))
    f_cat = float(composition.get("DOTAP", 0.0))
    structural = max(0.0, 1.0 - f_peg - f_chol - f_cat)

    conc = float(process.get("total_lipid_conc", 5.0))   # mg/mL
    flow = float(process.get("flow_ratio", 3.0))         # aqueous:organic

    # particle size (nm)
    size = 55.0 + 7.0 * conc + 55.0 / max(flow, 0.5) + 40.0 * f_chol + 90.0 * f_cat
    # polydispersity index
    pdi = 0.06 + 0.28 * ((flow - 3.0) / 3.0) ** 2 + 0.25 * f_cat
    # zeta potential (mV) — cationic raises it; PEG shields toward 0
    zeta = (60.0 * f_cat - 2.0) * math.exp(-6.0 * f_peg)
    # encapsulation efficiency (%)
    enc = 38.0 + 42.0 * f_chol + 18.0 * structural - 35.0 * f_peg + 0.9 * conc
    # cargo retention (%)
    ret = 50.0 + 46.0 * f_chol - 22.0 * f_peg
    # mucus penetration (%) — the star objective
    pen = 100.0 * _g(size, 100.0, 45.0) * _g(zeta, 0.0, 12.0) * _g(f_peg, 0.07, 0.045)
    # effective diffusion ratio (relative to buffer)
    eff = 0.02 + 0.9 * (pen / 100.0)

    out = {
        "mucus_penetration": pen,
        "eff_diffusion": eff,
        "size_nm": size,
        "pdi": pdi,
        "zeta_mv": zeta,
        "encapsulation_pct": enc,
        "cargo_retention": ret,
    }
    if noise and rng is not None:
        scales = {"mucus_penetration": 3.0, "eff_diffusion": 0.03, "size_nm": 4.0,
                  "pdi": 0.01, "zeta_mv": 1.5, "encapsulation_pct": 2.5, "cargo_retention": 2.5}
        out = {k: v + rng.normal(0.0, scales.get(k, 0.0)) for k, v in out.items()}
    # clip to sane ranges
    out["mucus_penetration"] = float(np.clip(out["mucus_penetration"], 0.0, 100.0))
    out["encapsulation_pct"] = float(np.clip(out["encapsulation_pct"], 0.0, 100.0))
    out["cargo_retention"] = float(np.clip(out["cargo_retention"], 0.0, 100.0))
    out["pdi"] = float(np.clip(out["pdi"], 0.01, 1.0))
    out["size_nm"] = float(max(20.0, out["size_nm"]))
    out["eff_diffusion"] = float(max(0.0, out["eff_diffusion"]))
    return {k: round(v, 4) for k, v in out.items()}


def make_demo_campaign() -> CampaignConfig:
    """A representative MPP optimization problem used by the seed script and tests."""
    return CampaignConfig(
        name="Demo: DSPC/Chol/PEG/DOTAP muco-penetrating liposome",
        description="Synthetic demo campaign — optimize mucus penetration & encapsulation, "
                    "target ~100 nm, keep PDI low and zeta near-neutral.",
        components=[
            ComponentSpec(lipid="DSPC", is_filler=True),               # structural remainder
            ComponentSpec(lipid="Cholesterol", low=0.10, high=0.50),
            ComponentSpec(lipid="DSPE-PEG2000", low=0.00, high=0.15),
            ComponentSpec(lipid="DOTAP", low=0.00, high=0.30),
        ],
        process_params=[
            ProcessParamSpec(name="total_lipid_conc", low=1.0, high=20.0, unit="mg/mL"),
            ProcessParamSpec(name="flow_ratio", low=1.0, high=5.0, unit="aq:org"),
        ],
        objectives=[
            ObjectiveSpec(readout="mucus_penetration", direction="max", weight=1.0),
            ObjectiveSpec(readout="encapsulation_pct", direction="max", weight=1.0),
            ObjectiveSpec(readout="size_nm", direction="target", target=100.0, weight=0.5),
        ],
        constraints=[
            ConstraintSpec(readout="pdi", op="<=", bound=0.3),
            ConstraintSpec(readout="zeta_mv", op="between", bound=-10.0, bound2=10.0),
        ],
    )
