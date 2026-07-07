"""Page 3 — enter measured readouts and attach raw files for a formulation."""
import streamlit as st

from mpp import service, storage
from mpp.config import STANDARD_READOUTS
from mpp.schema import ExperimentRecord
from mpp.ui import composition_summary, require_campaign

st.set_page_config(page_title="Upload Results · MPP", page_icon="🧫", layout="wide")
camp = require_campaign()
cfg = camp["config"]
st.title("Upload Results")
st.caption(f"Campaign: **{cfg.name}**")

READOUT_UNITS = {r[0]: r[1] for r in STANDARD_READOUTS}
ALL_READOUTS = [r[0] for r in STANDARD_READOUTS]
required = list(dict.fromkeys([o.readout for o in cfg.objectives] + [c.readout for c in cfg.constraints]))

# ---- selection widgets (outside the form so they update the form live) ----
mode = st.radio("Mode", ["Complete a planned experiment", "Add a new record"], horizontal=True)

prefill = {"composition": {}, "process": {}}
ctx = "new"
target_exp_id = None

if mode == "Complete a planned experiment":
    planned = service.list_experiments(camp["id"], status="suggested")
    if not planned:
        st.info("No planned experiments. Generate some on **Suggest Experiments**, or switch to *Add a new record*.")
        st.stop()
    options = {
        e["id"]: f"#{e['id']} · {e['well'] or '—'} · {composition_summary(e['composition'])}"
        for e in planned
    }
    target_exp_id = st.selectbox("Planned experiment", list(options), format_func=lambda i: options[i])
    chosen = next(e for e in planned if e["id"] == target_exp_id)
    prefill = {"composition": chosen["composition"], "process": chosen["process"]}
    ctx = f"exp{target_exp_id}"

extra = st.multiselect(
    "Readouts you are recording", ALL_READOUTS,
    default=[r for r in required if r in ALL_READOUTS] or ALL_READOUTS[:1],
    help="Required for this campaign's objectives/constraints are pre-selected.",
)

# ---- the data-entry form ----
with st.form(f"entry_{ctx}"):
    c1, c2, c3 = st.columns(3)
    label = c1.text_input("Label", value="")
    plate = c2.text_input("Plate", value=f"campaign{camp['id']}")
    well = c3.text_input("Well", value="")

    st.markdown("**Composition (mole fractions)**")
    comp_cols = st.columns(min(4, max(1, len(cfg.components))))
    composition = {}
    for idx, comp in enumerate(cfg.components):
        col = comp_cols[idx % len(comp_cols)]
        default = float(prefill["composition"].get(comp.lipid, 0.0))
        tag = " (filler)" if comp.is_filler else ""
        composition[comp.lipid] = col.number_input(
            f"{comp.lipid}{tag}", min_value=0.0, max_value=1.0, value=default, step=0.01,
            key=f"comp_{ctx}_{comp.lipid}",
        )

    process = {}
    if cfg.process_params:
        st.markdown("**Process parameters**")
        proc_cols = st.columns(min(4, len(cfg.process_params)))
        for idx, p in enumerate(cfg.process_params):
            col = proc_cols[idx % len(proc_cols)]
            default = float(prefill["process"].get(p.name, (p.low + p.high) / 2))
            process[p.name] = col.number_input(
                f"{p.name} ({p.unit})" if p.unit else p.name,
                value=default, key=f"proc_{ctx}_{p.name}",
            )

    st.markdown("**Measured readouts**")
    readouts = {}
    if extra:
        r_cols = st.columns(min(4, len(extra)))
        for idx, r in enumerate(extra):
            col = r_cols[idx % len(r_cols)]
            unit = READOUT_UNITS.get(r, "")
            readouts[r] = col.number_input(f"{r} ({unit})" if unit else r, value=0.0,
                                           key=f"ro_{ctx}_{r}")
    else:
        st.warning("Select at least one readout above.")

    notes = st.text_area("Notes", value="", height=70)
    files = st.file_uploader(
        "Attach raw files (Excel, PDF, images, plots, markdown, notes)",
        accept_multiple_files=True,
    )

    submitted = st.form_submit_button("💾 Save result", type="primary")

if submitted:
    comp_sum = sum(composition.values())
    if abs(comp_sum - 1.0) > 0.02:
        st.warning(f"Composition sums to {comp_sum:.3f} (not 1.0). Saved anyway — double-check the fractions.")
    if mode == "Complete a planned experiment":
        service.update_experiment(
            target_exp_id, composition=composition, process=process, readouts=readouts,
            label=label, plate=plate, well=well, notes=notes, status="completed",
        )
        exp_id = target_exp_id
    else:
        rec = ExperimentRecord(composition=composition, process=process, readouts=readouts,
                               label=label, plate=plate, well=well, notes=notes)
        exp_id = service.add_experiment(camp["id"], rec, source="manual", status="completed")

    saved_files = 0
    for f in files or []:
        meta = storage.save_upload(exp_id, f)
        service.add_attachment(exp_id, meta)
        saved_files += 1

    st.success(f"Saved experiment #{exp_id} with {saved_files} attachment(s).")
    st.toast("Result recorded — refit on the Model & Insights page.")
