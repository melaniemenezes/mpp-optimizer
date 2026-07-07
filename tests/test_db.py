"""DB/schema round-trip tests on an isolated temp database."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mpp.db import Attachment, Base, Campaign, Experiment
from mpp.schema import CampaignConfig, ComponentSpec, ExperimentRecord, ObjectiveSpec


def test_experiment_json_roundtrip(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, future=True)

    with Session() as s:
        c = Campaign(name="c1", config={"k": "v", "n": 3})
        s.add(c)
        s.flush()
        e = Experiment(
            campaign_id=c.id,
            composition={"DSPC": 0.7, "Cholesterol": 0.3},
            process={"flow_ratio": 3.0},
            readouts={"size_nm": 101.5, "pdi": 0.12},
        )
        s.add(e)
        s.flush()
        s.add(Attachment(experiment_id=e.id, filename="dls.pdf", path="/tmp/dls.pdf", size=42))
        s.commit()
        eid = e.id

    with Session() as s:
        e2 = s.get(Experiment, eid)
        assert e2.composition["DSPC"] == 0.7
        assert e2.readouts["size_nm"] == 101.5
        assert len(e2.attachments) == 1
        assert e2.attachments[0].filename == "dls.pdf"
        assert e2.campaign.config["n"] == 3


def test_campaign_config_validation():
    cfg = CampaignConfig(
        name="t",
        components=[ComponentSpec(lipid="DSPC", is_filler=True),
                    ComponentSpec(lipid="Cholesterol", low=0.1, high=0.5)],
        objectives=[ObjectiveSpec(readout="mucus_penetration", direction="max")],
    )
    assert cfg.filler().lipid == "DSPC"
    assert [c.lipid for c in cfg.free_components()] == ["Cholesterol"]
    assert cfg.dim_names() == ["x[Cholesterol]"]
    # round-trip through dict (as stored in the DB)
    cfg2 = CampaignConfig(**cfg.model_dump())
    assert cfg2.name == "t"


def test_experiment_record_defaults():
    rec = ExperimentRecord(composition={"DSPC": 1.0})
    assert rec.process == {}
    assert rec.readouts == {}
