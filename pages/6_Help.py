"""Page 6 — Help: the science, a walkthrough, the configured study, and an FAQ."""
import pandas as pd
import streamlit as st

from mpp.config import STANDARD_READOUTS
from mpp.ui import apply_theme, badges, ensure_init, page_header

st.set_page_config(page_title="Help · MPP", page_icon="🧫", layout="wide")
apply_theme()
ensure_init()
page_header("Help & Guide", "Everything you need to understand and use the MPP Optimizer.", icon="❓")

tab_overview, tab_science, tab_howto, tab_study, tab_faq = st.tabs(
    ["Overview", "The science", "Using the app", "Diffusion study", "FAQ"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    with st.container(border=True):
        st.subheader("What this app is")
        st.markdown(
            """
The **MPP Optimizer** helps you design **muco-penetrating liposomal nanoparticles (MPPs)** — tiny
lipid bubbles engineered to slip through mucus and deliver a drug. The recipe space is enormous
(dozens of lipids × continuous ratios × process settings → millions of formulations), and each
wet-lab test is slow, so the app uses **Bayesian optimization** (active learning): it learns from
the experiments you've done and proposes the ones most worth doing next.
            """
        )
        badges("Streamlit", "scikit-learn Gaussian Processes", "multi-objective", "SQLite", "no GPU needed")

    with st.container(border=True):
        st.subheader("The active-learning loop")
        st.markdown(
            """
1. **Define** a campaign (lipids, input features, objectives, constraints).
2. **Suggest** — the model proposes a batch of formulations to run.
3. **Run** them on the bench / 96-well platform.
4. **Upload** the measured results (+ attach raw files).
5. **Learn** — refit the model; it proposes better formulations. Repeat.
            """
        )

# ---------------------------------------------------------------- Science
with tab_science:
    with st.container(border=True):
        st.subheader("Why mucus is the problem")
        st.markdown(
            """
Mucus is a sticky, mesh-like gel that traps foreign particles — and it's thicker in respiratory
disease. A particle gets **through** it if it is **small enough** to fit the mesh and **not sticky**
(near-neutral surface charge, often with a PEG "stealth" coat). Those are **muco-penetrating**
particles (MPPs). The app optimizes the formulation to achieve exactly that.
            """
        )

    with st.container(border=True):
        st.subheader("The lipids (the recipe)")
        lip = pd.DataFrame(
            [
                ["Structural (HSPC, DSPC, DPPC…)", "The bilayer wall of the bubble."],
                ["Cholesterol", "Stiffens the wall, reduces leakiness."],
                ["PEG-lipid (mPEG, DSPE-PEG…)", "‘Stealth’ coat → muco-inert, slips through mucus."],
                ["Cationic (DDAB, DOTAP…)", "Positively charged."],
                ["Anionic (DSPG, DOPG…)", "Negatively charged."],
            ],
            columns=["Lipid type", "Role"],
        )
        st.dataframe(lip, hide_index=True, width="stretch")
        st.caption("A composition is the set of mole fractions (proportions) of the chosen lipids — they sum to 1.")

    with st.container(border=True):
        st.subheader("The readouts (what we measure)")
        st.markdown("**Particle attributes & transport**")
        rd = pd.DataFrame(
            [(name, unit or "—", direction, desc) for (name, unit, direction, desc) in STANDARD_READOUTS],
            columns=["Readout", "Unit", "Default goal", "Meaning"],
        )
        st.dataframe(rd, hide_index=True, width="stretch")
        st.markdown(
            """
**Reading the diffusion metrics (from particle tracking in mucin):**
- **D** — overall speed of spreading (higher = faster = better penetration).
- **D₁ / Dα** — the diffusion coefficient measured over a short (1 s) vs longer (10 s) window.
- **α (alpha)** — *the* mobility fingerprint: **≈1** = free diffusion, **<1** = hindered/**trapped**
  (subdiffusive), **>1** = directed.
- **net-to-path** — straight-line distance ÷ total wiggly distance: **→1** directed/mobile,
  **→0** confined (wiggles in place). Together with α it distinguishes *stuck* from *mobile* particles.
            """
        )

# ---------------------------------------------------------------- How-to
with tab_howto:
    steps = [
        ("🧪 Campaign Setup",
         "Pick the lipids (mark one as the structural/filler that absorbs the remaining fraction), set "
         "mole-fraction ranges, add process parameters / input features (e.g. size, zeta), and choose the "
         "objectives (max/min/target) and constraints. Or use a one-click ready-made campaign."),
        ("✨ Suggest Experiments",
         "Choose how many formulations to propose. With little data you get a space-filling design (seed a "
         "plate); with enough data, Bayesian suggestions. Export a 96-well worklist CSV, or save the batch as "
         "planned experiments to fill in later."),
        ("⬆️ Upload Results",
         "Complete a planned experiment or add a new record: enter the composition, input features, and measured "
         "readouts, and drag-and-drop the raw files (Excel, PDF, images, plots, notes). Files are stored and "
         "previewed — you confirm the key numbers, keeping the dataset clean."),
        ("🗂️ Dataset Browser",
         "The whole campaign as one standardized table (x: composition, p: inputs, y: readouts). Filter, export "
         "to CSV/Excel, and inspect any experiment's attachments."),
        ("📈 Model & Insights",
         "See the Pareto trade-off front and the recommended best compositions. Under ‘Characterise a readout’, "
         "pick any output (e.g. a diffusion metric) to see which inputs drive it, its partial-dependence curves, "
         "and a leave-one-out predicted-vs-observed fit."),
    ]
    for title, body in steps:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.markdown(body)

# ---------------------------------------------------------------- Study
with tab_study:
    with st.container(border=True):
        st.subheader("Configured study: mucin-diffusion characterisation")
        st.markdown(
            """
A ready-made campaign matches a specific study design — mapping formulation + physicochemical
properties to particle mobility in mucin (from multiple-particle tracking, MPT).
            """
        )
        st.markdown("**Input features (7)**")
        badges("DDAB", "DSPG", "HSPC (filler)", "Cholesterol", "mPEG", "liposome size (nm)", "zeta (mV)")
        st.markdown("**Diffusion outputs (5)**")
        badges("D (>5 s)", "D₁ (1 s)", "Dα (10 s)", "α exponent", "net-to-path")
        st.markdown(
            """
Create it from **Campaign Setup → Quick start → Create diffusion-study campaign**, or seed synthetic
data with `python scripts/seed_diffusion_demo.py`. Then, on **Model & Insights**, pick each diffusion
output to characterise how the inputs drive it.
            """
        )
        st.info(
            "Modelling note: because size and zeta are *consequences* of the formulation, using them as "
            "inputs suits a **characterisation** model. To *design* new formulations you'd additionally "
            "predict size/zeta from composition — a straightforward extension.",
            icon="💡",
        )

# ---------------------------------------------------------------- FAQ
with tab_faq:
    faqs = [
        ("Is this Bayesian optimization or a Bayesian network?",
         "Bayesian **optimization** — an active-learning loop that finds the best formulation. A Bayesian "
         "*network* (a graph of probabilistic dependencies for mechanistic interpretation) is a complementary "
         "idea that could be layered on the dataset later; it isn't what drives the search here."),
        ("Why scikit-learn Gaussian Processes instead of Ax/BoTorch?",
         "Modern Ax needs Python ≥3.10 and pulls in PyTorch. To stay lightweight and run anywhere (incl. Python "
         "3.9, no GPU), the engine is built on scikit-learn GPs + SciPy. Same behaviour for this scale: GP "
         "surrogate, multi-objective (ParEGO), constraints, Pareto front, sensitivities."),
        ("How does it suggest formulations before I have data?",
         "It uses a Latin-hypercube space-filling design to cover the recipe space broadly — ideal for seeding "
         "your first 96-well plate. Once enough experiments are completed, it switches to Bayesian suggestions."),
        ("Why is shadcn/ui not used natively?",
         "shadcn/ui is a React component library and this is a Python/Streamlit app, so its components can't be "
         "dropped in directly. The app instead adopts shadcn's design language (Inter, neutral palette, bordered "
         "cards) via a global theme."),
        ("Where is my data stored?",
         "Locally, in a single SQLite file (data/mpp.db) plus an attachments folder. No external services, no "
         "cloud. Export any time to CSV/Excel from the Dataset Browser."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(a)
