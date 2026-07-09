from pathlib import Path

from lottie.distill.io import load_spec, save_draft
from lottie.distill.schema import DistilledSkillSpec, DistillProvenance


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    spec = DistilledSkillSpec(name="summ", template="Summarize {topic}.", slots=["topic"])
    prov = DistillProvenance(source_agent="Digest", namespace="ns", source_run_ids=["r1", "r2"])
    d = save_draft(tmp_path, spec, prov)

    assert d == tmp_path / "skills" / "draft" / "summ"
    assert (d / "template.yaml").is_file()
    assert (d / "provenance.yaml").is_file()
    assert (d / "SKILL.md").is_file()
    # draft must NOT be a discoverable skill (no skill.py)
    assert not (d / "skill.py").exists()

    loaded = load_spec(d)
    assert loaded.name == "summ"
    assert loaded.template == "Summarize {topic}."
    assert loaded.slots == ["topic"]
    assert loaded.version == "0.1.0"


def test_template_input_output_defaults() -> None:
    from lottie.distill.schema import TemplateInput, TemplateOutput

    assert TemplateInput().slots == {}
    assert TemplateOutput(content="x").content == "x"
