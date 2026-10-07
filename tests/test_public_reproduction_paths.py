import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_formal_configs_and_public_figure_source_exist() -> None:
    required = [
        ROOT / "configs/ai_darld_v3.yaml",
        ROOT / "configs/mechanism_confirmation_v3.yaml",
        ROOT / "configs/full_scale.yaml",
        ROOT / "configs/evidence_increment_synthesis.json",
        ROOT
        / "outputs/ai_darld_v3/evidence_increment_synthesis/product_level_cost_inputs.csv",
    ]
    assert all(path.is_file() for path in required)


def test_formal_configuration_identity() -> None:
    v3 = yaml.safe_load((ROOT / "configs/ai_darld_v3.yaml").read_text(encoding="utf-8"))
    retail = yaml.safe_load(
        (ROOT / "configs/mechanism_confirmation_v3.yaml").read_text(encoding="utf-8")
    )
    assert v3["factorized_transfer"]["fixed_anchor_candidates"] == 4
    assert v3["factorized_transfer"]["random_candidate_count"] == 96
    assert retail["uci_confirmation_v3"]["confirmation_products"] == 120


def test_default_figure_config_uses_released_source() -> None:
    config = json.loads(
        (ROOT / "configs/evidence_increment_synthesis.json").read_text(encoding="utf-8")
    )
    compact_source = ROOT / config["compact_source"]
    assert compact_source.is_file()
    assert "outputs/runs" not in compact_source.as_posix()
