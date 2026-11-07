import pyrallis

from piadaptwin.manifold import ExperimentConfig


def test_main_config_loads() -> None:
    cfg = pyrallis.parse(
        config_class=ExperimentConfig, args=["--config_path", "configs/experiment/main.yaml"]
    )
    assert cfg.name == "main"
    assert cfg.data.source == "vitaldb"
    assert cfg.flow.grid_points == 12


def test_ablation_flags_apply() -> None:
    cfg = pyrallis.parse(
        config_class=ExperimentConfig,
        args=["--config_path", "configs/experiment/ablation_no_wk.yaml"],
    )
    assert cfg.use_windkessel is False
    assert cfg.use_pkpd is True


def test_dump_and_reload_preserves_name() -> None:
    cfg = ExperimentConfig(name="probe")
    text = pyrallis.dump(cfg)
    assert "probe" in text
