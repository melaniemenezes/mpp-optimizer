"""Canonical, validated data contract for the standardized dataset.

These pydantic models define what a *campaign* (the optimization problem) and an
*experiment record* (one formulation + its measured readouts) look like. The DB
stores the campaign config and the per-experiment composition/process/readouts as
JSON; these models are the validated bridge between the DB and the rest of the app.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Direction = Literal["max", "min", "target"]
ConstraintOp = Literal["<=", ">=", "between"]


class ComponentSpec(BaseModel):
    """One lipid component of the formulation and the mole-fraction range to explore."""

    lipid: str
    low: float = 0.0
    high: float = 1.0
    # The filler/structural lipid absorbs the remainder so fractions sum to 1.
    # Exactly one component should be the filler.
    is_filler: bool = False

    @field_validator("low", "high")
    @classmethod
    def _frac_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("mole fraction bounds must be within [0, 1]")
        return v


class ProcessParamSpec(BaseModel):
    """A continuous process parameter of the liquid-assembly platform."""

    name: str
    low: float
    high: float
    unit: str = ""


class ObjectiveSpec(BaseModel):
    """A readout to optimize."""

    readout: str
    direction: Direction = "max"
    target: Optional[float] = None  # required when direction == "target"
    weight: float = 1.0


class ConstraintSpec(BaseModel):
    """A hard requirement on a readout (handled as a probabilistic feasibility constraint)."""

    readout: str
    op: ConstraintOp
    bound: float
    bound2: Optional[float] = None  # upper bound when op == "between"


class CampaignConfig(BaseModel):
    """The full definition of an optimization campaign."""

    name: str
    description: str = ""
    components: List[ComponentSpec]
    process_params: List[ProcessParamSpec] = Field(default_factory=list)
    objectives: List[ObjectiveSpec]
    constraints: List[ConstraintSpec] = Field(default_factory=list)

    # --- convenience accessors used by the optimizer -----------------------
    def free_components(self) -> List[ComponentSpec]:
        return [c for c in self.components if not c.is_filler]

    def filler(self) -> Optional[ComponentSpec]:
        for c in self.components:
            if c.is_filler:
                return c
        return None

    def dim_names(self) -> List[str]:
        """Ordered names of the optimization dimensions (free lipid fractions + process params)."""
        names = [f"x[{c.lipid}]" for c in self.free_components()]
        names += [p.name for p in self.process_params]
        return names

    def bounds(self):
        """(low, high) per optimization dimension, matching dim_names() order."""
        lows, highs = [], []
        for c in self.free_components():
            lows.append(c.low)
            highs.append(c.high)
        for p in self.process_params:
            lows.append(p.low)
            highs.append(p.high)
        return lows, highs


class ExperimentRecord(BaseModel):
    """One formulation and (optionally) its measured readouts."""

    composition: Dict[str, float]
    process: Dict[str, float] = Field(default_factory=dict)
    readouts: Dict[str, float] = Field(default_factory=dict)
    label: str = ""
    plate: str = ""
    well: str = ""
    notes: str = ""
