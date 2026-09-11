"""PipelineConfig round-trips: every dataclass field survives INI and YAML."""

import dataclasses
import typing

import pytest

from pyrt_transient.config_trans import PipelineConfig


def _mutate_everything(config: PipelineConfig) -> PipelineConfig:
    """Set every field of every section to a non-default value."""
    for section_name in config._SECTIONS:
        section = getattr(config, section_name)
        for f in dataclasses.fields(section):
            cur = getattr(section, f.name)
            if isinstance(cur, bool):
                setattr(section, f.name, not cur)
            elif isinstance(cur, int):
                setattr(section, f.name, cur + 7)
            elif isinstance(cur, float):
                setattr(section, f.name, cur * 1.5 + 0.25)
            elif isinstance(cur, str):
                setattr(section, f.name, cur + "_x")
            elif isinstance(cur, list):
                setattr(section, f.name, ["alpha", "beta"])
            elif isinstance(cur, dict):
                # from_dict *merges* logging.module_levels (documented), so
                # extend the existing dict rather than replacing it.
                setattr(section, f.name, {**cur, "k1": 1.5, "k2": 2.5} if section_name == "detection"
                        else {**cur, "m": "DEBUG"})
            elif cur is None:
                # Respect the declared type: Optional[float] fields that
                # default to None must round-trip as floats.
                inner = [a for a in typing.get_args(f.type) if a is not type(None)]
                setattr(section, f.name, 0.75 if inner and inner[0] is float else "set_from_none")
    config.base_data_dir = "/tmp/elsewhere"
    config.base_public_dir = "/tmp/public"
    config.generate_frontend = not config.generate_frontend
    config.max_workers = 11
    return config


def _assert_equal_configs(a: PipelineConfig, b: PipelineConfig):
    for section_name in a._SECTIONS:
        sa, sb = getattr(a, section_name), getattr(b, section_name)
        for f in dataclasses.fields(sa):
            va, vb = getattr(sa, f.name), getattr(sb, f.name)
            if isinstance(va, float):
                assert vb == pytest.approx(va), f"{section_name}.{f.name}"
            else:
                assert va == vb, f"{section_name}.{f.name}: {va!r} != {vb!r}"
    for name in ("base_data_dir", "base_public_dir", "generate_frontend", "max_workers"):
        assert getattr(a, name) == getattr(b, name), name


def test_ini_round_trip_covers_every_field(tmp_path):
    """The old hand-listed INI loader silently dropped siglim,
    new_source_siglim, catalogs, maglim_filter_multiplier and more."""
    config = _mutate_everything(PipelineConfig())
    path = tmp_path / "c.ini"
    config.to_file(str(path))
    loaded = PipelineConfig.from_file(str(path))
    _assert_equal_configs(config, loaded)
    # The ones that used to be lost, explicitly:
    assert loaded.detection.siglim == config.detection.siglim
    assert loaded.detection.new_source_siglim == config.detection.new_source_siglim
    assert loaded.detection.catalogs == ["alpha", "beta"]
    assert loaded.detection.maglim_filter_multiplier == config.detection.maglim_filter_multiplier


def test_ini_accepts_comma_separated_lists_and_ignores_bad_values(tmp_path):
    path = tmp_path / "c.ini"
    path.write_text("[detection]\ncatalogs = gaia, usno\nsiglim = not-a-number\nmin_n_detections = 4\n")
    loaded = PipelineConfig.from_file(str(path))
    assert loaded.detection.catalogs == ["gaia", "usno"]
    assert loaded.detection.siglim == PipelineConfig().detection.siglim  # bad value ignored
    assert loaded.detection.min_n_detections == 4


def test_yaml_dict_round_trip_matches_ini(tmp_path):
    config = _mutate_everything(PipelineConfig())
    as_dict = {name: dataclasses.asdict(getattr(config, name)) for name in config._SECTIONS}
    as_dict["global"] = {"base_data_dir": config.base_data_dir, "base_public_dir": config.base_public_dir,
                         "generate_frontend": config.generate_frontend, "max_workers": config.max_workers}
    loaded = PipelineConfig.from_dict(as_dict)
    _assert_equal_configs(config, loaded)


def test_ini_round_trip_preserves_explicit_none(tmp_path):
    """Optional fields with a NON-None default: writing them as an absent key
    restored the default on read, so `to_file` then `from_file` silently
    turned an explicit "no gate here" into siglim=1.5."""
    config = PipelineConfig()
    assert config.detection.new_source_siglim is not None, "guards the premise"
    assert config.detection.unphotometered_veto_max_brightening_mag is not None
    config.detection.new_source_siglim = None
    config.detection.unphotometered_veto_max_brightening_mag = None
    path = tmp_path / "c.ini"
    config.to_file(str(path))
    loaded = PipelineConfig.from_file(str(path))
    assert loaded.detection.new_source_siglim is None
    assert loaded.detection.unphotometered_veto_max_brightening_mag is None
