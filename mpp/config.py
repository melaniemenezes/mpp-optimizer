"""Paths and app-wide constants."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ATTACH_DIR = DATA_DIR / "attachments"
DB_PATH = DATA_DIR / "mpp.db"
DB_URL = f"sqlite:///{DB_PATH}"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ATTACH_DIR.mkdir(parents=True, exist_ok=True)

# Below which number of completed experiments the optimizer falls back to a
# space-filling design instead of fitting a GP (cold start).
MIN_POINTS_FOR_MODEL = 6

# Standard critical quality attributes (CQAs) / readouts a user can pick from
# when defining objectives and constraints. (name, unit, default_direction, help)
STANDARD_READOUTS = [
    ("mucus_penetration", "%", "max", "Fraction of particles that traverse the mucus layer (transport assay)."),
    ("eff_diffusion", "x", "max", "Effective diffusion coefficient relative to buffer (MSD-based)."),
    ("size_nm", "nm", "target", "Z-average hydrodynamic diameter."),
    ("pdi", "", "min", "Polydispersity index (lower = more monodisperse)."),
    ("zeta_mv", "mV", "target", "Zeta potential; near 0 mV favours muco-inert behaviour."),
    ("encapsulation_pct", "%", "max", "Encapsulation efficiency of the cargo."),
    ("cargo_retention", "%", "max", "Cargo retained after incubation (stability)."),
]

# File-type buckets for attachment preview.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv"}
PDF_EXTS = {".pdf"}
EXCEL_EXTS = {".xlsx", ".xls"}
