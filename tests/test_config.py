from cold_start_replenishment.config import load_audit_config


def test_load_audit_config() -> None:
    config = load_audit_config()
    assert config.random_seed == 20260805
    assert config.m5.filenames["calendar"] == "calendar.csv"
