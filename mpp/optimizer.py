"""Multi-objective Bayesian optimization engine (scikit-learn Gaussian Processes).

Design (lightweight, dependency-robust alternative to Ax/BoTorch):
  * Surrogate     : sklearn GaussianProcessRegressor with an ARD-RBF kernel.
  * Multi-objective: ParEGO — random augmented-Tchebycheff scalarizations, one GP
                     per batch member, giving a spread of Pareto-diverse suggestions.
  * Acquisition   : Expected Improvement, maximized over a large candidate pool.
  * Constraints   : constrained EI — EI is multiplied by the GP-estimated probability
                     that each outcome constraint is satisfied (near-neutral zeta, etc.).
  * Cold start    : Latin-hypercube space-filling design until enough data exists.
  * Interpretability: ARD lengthscales -> relative importance; partial-dependence curves.

The optimizer is stateless: it is rebuilt from the stored experiment records on every
call, so the SQLite dataset stays the single source of truth.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from . import config
from .schema import CampaignConfig, ObjectiveSpec

Record = Dict[str, dict]  # {"composition": {...}, "process": {...}, "readouts": {...}}


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class MPPOptimizer:
    def __init__(self, campaign: Union[CampaignConfig, dict], rng_seed: int = 0):
        self.config = campaign if isinstance(campaign, CampaignConfig) else CampaignConfig(**campaign)
        self.free = self.config.free_components()
        self.filler = self.config.filler()
        self.process_params = self.config.process_params
        self.objectives = self.config.objectives
        self.n_free = len(self.free)
        lows, highs = self.config.bounds()
        self.lows = np.array(lows, dtype=float)
        self.highs = np.array(highs, dtype=float)
        self.dim = len(self.lows)
        self.dim_names = self.config.dim_names()
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------ encode
    def _encode(self, rec: Record) -> np.ndarray:
        x = []
        comp = rec.get("composition", {}) or {}
        proc = rec.get("process", {}) or {}
        for c in self.free:
            x.append(_num(comp.get(c.lipid)) or 0.0)
        for p in self.process_params:
            v = _num(proc.get(p.name))
            x.append(v if v is not None else (p.low + p.high) / 2.0)
        return np.array(x, dtype=float)

    def _decode(self, x: np.ndarray) -> Record:
        # Round free fractions first, then let the filler absorb the remainder so the
        # composition sums to exactly 1 after rounding.
        comp: Dict[str, float] = {}
        s = 0.0
        for i, c in enumerate(self.free):
            v = round(max(0.0, float(x[i])), 4)
            comp[c.lipid] = v
            s += v
        if self.filler is not None:
            comp[self.filler.lipid] = round(max(0.0, 1.0 - s), 4)
        elif s > 0:  # pure mixture: renormalize to sum 1
            comp = {k: round(v / s, 4) for k, v in comp.items()}
        proc = {p.name: round(float(x[self.n_free + j]), 4) for j, p in enumerate(self.process_params)}
        return {"composition": comp, "process": proc}

    # ------------------------------------------------------------------ scaling
    def _to_unit(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lows) / (self.highs - self.lows + 1e-12)

    def _from_unit(self, U: np.ndarray) -> np.ndarray:
        return self.lows + U * (self.highs - self.lows)

    def _enforce_simplex(self, X: np.ndarray) -> np.ndarray:
        """Scale free-lipid fractions down so they sum to <= 1 (only if a filler exists)."""
        if self.filler is None or self.n_free == 0:
            return X
        s = X[:, : self.n_free].sum(axis=1, keepdims=True)
        scale = np.where(s > 1.0, (1.0 - 1e-3) / (s + 1e-12), 1.0)
        X = X.copy()
        X[:, : self.n_free] *= scale
        return X

    # ------------------------------------------------------------------ sampling
    def _space_filling(self, n: int) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dim, seed=int(self.rng.integers(1, 2**31 - 1)))
        U = sampler.random(n)
        return self._enforce_simplex(self._from_unit(U))

    def _candidates(self, n: int) -> np.ndarray:
        U = self.rng.random((n, self.dim))
        return self._enforce_simplex(self._from_unit(U))

    # ------------------------------------------------------------------ GP
    def _fit_gp(self, U: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        d = U.shape[1]
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[0.3] * d, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(1e-3, (1e-6, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            alpha=1e-8,
            random_state=int(self.rng.integers(1, 2**31 - 1)),
        )
        # A lengthscale resting on a bound just means a dimension is ~flat — benign here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp.fit(U, y)
        return gp

    @staticmethod
    def _ei_min(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        """Expected improvement for a MINIMIZATION target."""
        sigma = np.maximum(sigma, 1e-9)
        imp = best - mu
        z = imp / sigma
        return imp * norm.cdf(z) + sigma * norm.pdf(z)

    # ------------------------------------------------------------------ objectives
    def _obj_loss(self, o: ObjectiveSpec, v: Optional[float]) -> float:
        if v is None:
            return np.nan
        if o.direction == "max":
            return -v
        if o.direction == "min":
            return v
        t = o.target if o.target is not None else 0.0
        return abs(v - t)

    def _loss_matrix(self, records: List[Record]) -> np.ndarray:
        L = np.zeros((len(records), len(self.objectives)))
        for i, r in enumerate(records):
            ro = r.get("readouts", {}) or {}
            for j, o in enumerate(self.objectives):
                L[i, j] = self._obj_loss(o, _num(ro.get(o.readout)))
        return L

    @staticmethod
    def _normalize(L: np.ndarray) -> np.ndarray:
        lo = np.nanmin(L, axis=0)
        hi = np.nanmax(L, axis=0)
        rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return (L - lo) / rng

    def is_complete(self, r: Record) -> bool:
        ro = r.get("readouts", {}) or {}
        return all(_num(ro.get(o.readout)) is not None for o in self.objectives)

    # ------------------------------------------------------------------ constraints
    def _feasible_mask(self, records: List[Record]) -> np.ndarray:
        mask = np.ones(len(records), dtype=bool)
        for i, r in enumerate(records):
            ro = r.get("readouts", {}) or {}
            for c in self.config.constraints:
                v = _num(ro.get(c.readout))
                if v is None:
                    continue
                if c.op == "<=" and not v <= c.bound:
                    mask[i] = False
                elif c.op == ">=" and not v >= c.bound:
                    mask[i] = False
                elif c.op == "between":
                    hi = c.bound2 if c.bound2 is not None else c.bound
                    if not (c.bound <= v <= hi):
                        mask[i] = False
        return mask

    def _feasibility_prob(self, candU: np.ndarray, Uobs: np.ndarray, records: List[Record]) -> np.ndarray:
        prob = np.ones(len(candU))
        for c in self.config.constraints:
            y = np.array([_num((r.get("readouts", {}) or {}).get(c.readout)) for r in records], dtype=object)
            finite = np.array([v is not None for v in y])
            if finite.sum() < 3:  # too little data to model this constraint reliably
                continue
            yv = np.array([float(v) for v in y[finite]])
            gp = self._fit_gp(Uobs[finite], yv)
            mu, sd = gp.predict(candU, return_std=True)
            sd = np.maximum(sd, 1e-9)
            if c.op == "<=":
                p = norm.cdf((c.bound - mu) / sd)
            elif c.op == ">=":
                p = 1.0 - norm.cdf((c.bound - mu) / sd)
            else:
                hi = c.bound2 if c.bound2 is not None else c.bound
                p = norm.cdf((hi - mu) / sd) - norm.cdf((c.bound - mu) / sd)
            prob = prob * np.clip(p, 0.0, 1.0)
        return prob

    # ------------------------------------------------------------------ public API
    def suggest_batch(self, records: Sequence[Record], n: int = 12, n_candidates: int = 4000) -> List[Record]:
        """Propose `n` formulations to run next.

        Cold start (few/no completed records) -> space-filling design.
        Otherwise -> constrained-EI / ParEGO Bayesian suggestions.
        Each returned record carries a `_method` tag.
        """
        completed = [r for r in records if self.is_complete(r)]
        if len(completed) < config.MIN_POINTS_FOR_MODEL or self.dim == 0:
            X = self._space_filling(max(1, n))
            return [{**self._decode(x), "_method": "space-filling"} for x in X]

        Uobs = self._to_unit(np.array([self._encode(r) for r in completed]))
        Ln = self._normalize(self._loss_matrix(completed))
        n_obj = Ln.shape[1]
        cand = self._candidates(n_candidates)
        candU = self._to_unit(cand)
        feas = self._feasibility_prob(candU, Uobs, completed)

        chosen_idx: List[int] = []
        rho = 0.05
        length = 0.07  # diversity kernel width in unit space
        for _ in range(n):
            w = self.rng.random(n_obj)
            w = w / w.sum()
            g = (Ln * w).max(axis=1) + rho * (Ln * w).sum(axis=1)  # Tchebycheff scalar (minimize)
            gp = self._fit_gp(Uobs, g)
            mu, sd = gp.predict(candU, return_std=True)
            ei = self._ei_min(mu, sd, float(g.min()))
            score = ei * feas
            for ci in chosen_idx:  # penalize closeness to already-chosen points
                d2 = np.sum((candU - candU[ci]) ** 2, axis=1)
                score = score * (1.0 - np.exp(-d2 / (2 * length ** 2)))
            chosen_idx.append(int(np.argmax(score)))
        return [{**self._decode(cand[i]), "_method": "bayesopt"} for i in chosen_idx]

    def _nondominated(self, L: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        n = len(L)
        nd = np.zeros(n, dtype=bool)
        for i in range(n):
            if not eligible[i]:
                continue
            dominated = False
            for j in range(n):
                if i == j or not eligible[j]:
                    continue
                if np.all(L[j] <= L[i]) and np.any(L[j] < L[i]):
                    dominated = True
                    break
            nd[i] = not dominated
        return nd

    def pareto(self, records: Sequence[Record]) -> dict:
        completed = [r for r in records if self.is_complete(r)]
        if not completed:
            return {"records": [], "is_pareto": [], "values": []}
        L = self._loss_matrix(completed)
        feas = self._feasible_mask(completed)
        nd = self._nondominated(L, feas)
        values = []
        for r in completed:
            ro = r.get("readouts", {}) or {}
            values.append({o.readout: _num(ro.get(o.readout)) for o in self.objectives})
        return {"records": completed, "is_pareto": nd.tolist(), "feasible": feas.tolist(), "values": values}

    def recommend(self, records: Sequence[Record], k: int = 5) -> List[Record]:
        completed = [r for r in records if self.is_complete(r)]
        if not completed:
            return []
        Ln = self._normalize(self._loss_matrix(completed))
        feas = self._feasible_mask(completed)
        w = np.array([o.weight for o in self.objectives], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        score = (Ln * w).sum(axis=1)
        score = np.where(feas, score, score + 1e3)  # push infeasible to the back
        order = np.argsort(score)
        return [completed[i] for i in order[:k]]

    # --------------------------------------------------------- interpretability
    def _lengthscales(self, gp: GaussianProcessRegressor) -> np.ndarray:
        def find_rbf(kern):
            if isinstance(kern, RBF):
                return kern.length_scale
            for attr in ("k1", "k2"):
                if hasattr(kern, attr):
                    r = find_rbf(getattr(kern, attr))
                    if r is not None:
                        return r
            return None

        ls = np.atleast_1d(find_rbf(gp.kernel_)).astype(float)
        if ls.size == 1:
            ls = np.repeat(ls, self.dim)
        return ls

    def _records_with_readout(self, records: Sequence[Record], key: str) -> List[Record]:
        return [r for r in records if _num((r.get("readouts", {}) or {}).get(key)) is not None]

    def available_readouts(self, records: Sequence[Record]) -> List[str]:
        """Readout keys present (numeric) in the data — objectives first, then the rest."""
        keys = set()
        for r in records:
            for k, v in (r.get("readouts", {}) or {}).items():
                if _num(v) is not None:
                    keys.add(k)
        obj = [o.readout for o in self.objectives]
        rest = sorted(k for k in keys if k not in obj)
        return [k for k in obj if k in keys] + rest

    def _default_readout(self, readout: Optional[str]) -> Optional[str]:
        if readout:
            return readout
        return self.objectives[0].readout if self.objectives else None

    def sensitivities(self, records: Sequence[Record], readout: Optional[str] = None) -> dict:
        """Which input factors drive a readout (ARD importances) + partial-dependence curves.

        `readout` selects the output to characterise; defaults to the first objective.
        Curves and importances are in the readout's own units (raw predicted value).
        """
        target = self._default_readout(readout)
        completed = self._records_with_readout(records, target) if target else []
        if target is None or len(completed) < config.MIN_POINTS_FOR_MODEL or self.dim == 0:
            return {"importances": {}, "partial_dependence": {}, "readout": target,
                    "note": "Need more completed experiments to fit the interpretability model."}
        Uobs = self._to_unit(np.array([self._encode(r) for r in completed]))
        y = np.array([_num((r.get("readouts", {}) or {}).get(target)) for r in completed], dtype=float)
        gp = self._fit_gp(Uobs, y)
        ls = self._lengthscales(gp)
        inv = 1.0 / np.maximum(ls, 1e-6)
        imp = inv / inv.sum()
        importances = {self.dim_names[i]: float(imp[i]) for i in range(self.dim)}

        pd = {}
        base = Uobs.mean(axis=0)
        grid = np.linspace(0.0, 1.0, 30)
        for d in np.argsort(imp)[::-1][: min(4, self.dim)]:
            Ug = np.tile(base, (30, 1))
            Ug[:, d] = grid
            mu = gp.predict(Ug)
            xvals = self.lows[d] + grid * (self.highs[d] - self.lows[d])
            pd[self.dim_names[d]] = {"x": xvals.tolist(), "y": mu.tolist()}  # raw readout units
        return {"importances": importances, "partial_dependence": pd, "readout": target}

    def predicted_vs_observed(self, records: Sequence[Record], readout: Optional[str] = None) -> dict:
        target = self._default_readout(readout)
        completed = self._records_with_readout(records, target) if target else []
        if target is None or len(completed) < config.MIN_POINTS_FOR_MODEL:
            return {"obs": [], "pred": [], "readout": target}
        Uobs = self._to_unit(np.array([self._encode(r) for r in completed]))
        y = np.array([_num((r.get("readouts", {}) or {}).get(target)) for r in completed], dtype=float)
        preds = np.zeros(len(y))
        for i in range(len(y)):  # leave-one-out
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            gp = self._fit_gp(Uobs[mask], y[mask])
            preds[i] = gp.predict(Uobs[i : i + 1])[0]
        return {"obs": y.tolist(), "pred": preds.tolist(), "readout": target}


def build_optimizer(campaign: Union[CampaignConfig, dict], rng_seed: int = 0) -> MPPOptimizer:
    return MPPOptimizer(campaign, rng_seed=rng_seed)
