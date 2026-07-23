"""Seed library of lipids (a representative subset of the ~80 candidate lipids).

Each entry has a short name (used everywhere as the key), a category that drives
its typical role in a formulation, and a full chemical name for reference.
"""
from __future__ import annotations

from typing import List, TypedDict


class LipidDef(TypedDict):
    name: str
    category: str
    full_name: str
    notes: str


# Categories: structural | cholesterol | peg | helper | cationic | anionic | ionizable
SEED_LIPIDS: List[LipidDef] = [
    # --- structural (bilayer-forming) phospholipids ---
    {"name": "DSPC", "category": "structural", "full_name": "1,2-distearoyl-sn-glycero-3-phosphocholine", "notes": "Saturated, high Tm; rigid bilayer."},
    {"name": "DPPC", "category": "structural", "full_name": "1,2-dipalmitoyl-sn-glycero-3-phosphocholine", "notes": "Saturated; Tm ~41C."},
    {"name": "DOPC", "category": "structural", "full_name": "1,2-dioleoyl-sn-glycero-3-phosphocholine", "notes": "Unsaturated; fluid bilayer."},
    {"name": "POPC", "category": "structural", "full_name": "1-palmitoyl-2-oleoyl-glycero-3-phosphocholine", "notes": "Mixed chain; fluid at RT."},
    {"name": "HSPC", "category": "structural", "full_name": "Hydrogenated soy phosphatidylcholine", "notes": "Used in Doxil-type liposomes."},
    {"name": "DMPC", "category": "structural", "full_name": "1,2-dimyristoyl-sn-glycero-3-phosphocholine", "notes": "Shorter chain; Tm ~24C."},
    {"name": "EPC", "category": "structural", "full_name": "Egg L-alpha-phosphatidylcholine", "notes": "Natural PC mixture."},
    # --- cholesterol ---
    {"name": "Cholesterol", "category": "cholesterol", "full_name": "Cholesterol", "notes": "Stiffens bilayer, reduces leakiness."},
    # --- PEG-lipids (stealth / muco-inert coating) ---
    {"name": "DSPE-PEG2000", "category": "peg", "full_name": "DSPE-N-[methoxy(polyethylene glycol)-2000]", "notes": "Standard stealth PEG-lipid."},
    {"name": "DMG-PEG2000", "category": "peg", "full_name": "1,2-dimyristoyl-rac-glycero-3-PEG2000", "notes": "Shed-able PEG used in LNPs."},
    {"name": "DSPE-PEG5000", "category": "peg", "full_name": "DSPE-PEG5000", "notes": "Longer PEG; denser brush."},
    {"name": "DSPE-PEG1000", "category": "peg", "full_name": "DSPE-PEG1000", "notes": "Shorter PEG."},
    {"name": "mPEG", "category": "peg", "full_name": "Methoxy-poly(ethylene glycol) lipid (mPEG-lipid)", "notes": "Stealth PEG coating; muco-inert surface."},
    # --- helper / fusogenic ---
    {"name": "DOPE", "category": "helper", "full_name": "1,2-dioleoyl-sn-glycero-3-phosphoethanolamine", "notes": "Fusogenic helper lipid."},
    {"name": "DSPE", "category": "helper", "full_name": "1,2-distearoyl-sn-glycero-3-phosphoethanolamine", "notes": "PE helper lipid."},
    # --- cationic ---
    {"name": "DOTAP", "category": "cationic", "full_name": "1,2-dioleoyl-3-trimethylammonium-propane", "notes": "Permanently cationic."},
    {"name": "DOTMA", "category": "cationic", "full_name": "1,2-di-O-octadecenyl-3-trimethylammonium propane", "notes": "Cationic transfection lipid."},
    {"name": "DC-Chol", "category": "cationic", "full_name": "3-beta-[N-(dimethylaminoethane)-carbamoyl]cholesterol", "notes": "Cationic cholesterol derivative."},
    {"name": "DDAB", "category": "cationic", "full_name": "Dimethyldioctadecylammonium bromide", "notes": "Cationic surfactant lipid."},
    # --- anionic ---
    {"name": "DOPG", "category": "anionic", "full_name": "1,2-dioleoyl-sn-glycero-3-phospho-(1'-rac-glycerol)", "notes": "Anionic PG lipid."},
    {"name": "DPPG", "category": "anionic", "full_name": "1,2-dipalmitoyl-sn-glycero-3-phospho-(1'-rac-glycerol)", "notes": "Saturated anionic PG."},
    {"name": "DSPG", "category": "anionic", "full_name": "1,2-distearoyl-sn-glycero-3-phospho-(1'-rac-glycerol)", "notes": "Saturated anionic PG."},
    # --- ionizable (LNP) ---
    {"name": "DLin-MC3-DMA", "category": "ionizable", "full_name": "Dilinoleylmethyl-4-dimethylaminobutyrate", "notes": "Ionizable lipid (Onpattro)."},
    {"name": "ALC-0315", "category": "ionizable", "full_name": "ALC-0315 ionizable lipid", "notes": "Comirnaty ionizable lipid."},
    {"name": "SM-102", "category": "ionizable", "full_name": "SM-102 ionizable lipid", "notes": "Spikevax ionizable lipid."},
]

LIPID_NAMES = [l["name"] for l in SEED_LIPIDS]
