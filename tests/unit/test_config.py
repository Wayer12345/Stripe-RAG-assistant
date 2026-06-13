"""Unit tests for ``app/utils/config.py``.

All tests use temporary directories and write a local ``config.yaml`` fixture.
No real project artifacts or external services are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from app.utils.config import EvalSettings, LocalSettings, Settings, load_settings
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: object) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _base_config() -> dict[str, object]:
    return {
        "app": {
            "name": "test-app",
            "environment": "test",
            "log_level": "DEBUG",
        },
        "paths": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "indexes_dir": "data/indexes",
            "manifests_dir": "data/manifests",
            "eval_dir": "eval",
        },
        "ingestion": {
            "input_dir": "data/raw",
            "recursive": True,
            "supported_extensions": [".txt", ".md"],
            "outputs": {
                "parsed_documents_path": "data/interim/parsed.jsonl",
                "cleaned_documents_path": "data/interim/cleaned.jsonl",
                "manifest_path": "data/manifests/manifest.json",
            },
        },
        "cleaning": {
            "mode": "conservative",
            "boilerplate": {
                "phrases": ["Sign in", "Log in"],
            },
        },
    }


def _make_valid_configs(config_dir: Path) -> None:
    """Write a complete valid minimal ``config.yaml`` to *config_dir*."""
    _write_yaml(config_dir / "config.yaml", _base_config())


# ---------------------------------------------------------------------------
# Happy-path loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadSettingsValid:
    def test_returns_settings_instance(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        result = load_settings(tmp_path)
        assert isinstance(result, Settings)

    def test_app_fields_loaded(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.app.name == "test-app"
        assert s.app.environment == "test"
        assert s.app.log_level == "DEBUG"

    def test_paths_are_path_objects(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert isinstance(s.paths.raw_dir, Path)
        assert isinstance(s.paths.interim_dir, Path)
        assert isinstance(s.paths.manifests_dir, Path)

    def test_ingestion_input_dir_is_path(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert isinstance(s.ingestion.input_dir, Path)

    def test_output_paths_are_path_objects(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert isinstance(s.ingestion.outputs.parsed_documents_path, Path)
        assert isinstance(s.ingestion.outputs.cleaned_documents_path, Path)
        assert isinstance(s.ingestion.outputs.manifest_path, Path)

    def test_eval_and_local_sections_use_typed_models(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert isinstance(s.eval, EvalSettings)
        assert isinstance(s.local, LocalSettings)

    def test_extensions_normalized_to_lowercase(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        cfg = _base_config()
        cfg["ingestion"] = {
            "input_dir": "data/raw",
            "recursive": True,
            "supported_extensions": [".TXT", ".MD", ".HTML"],
            "outputs": {
                "parsed_documents_path": "data/interim/parsed.jsonl",
                "cleaned_documents_path": "data/interim/cleaned.jsonl",
                "manifest_path": "data/manifests/manifest.json",
            },
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        s = load_settings(tmp_path)
        assert s.ingestion.supported_extensions == [".txt", ".md", ".html"]

    def test_cleaning_mode_conservative(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.cleaning.mode == "conservative"

    def test_cleaning_boilerplate_phrases_loaded(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert "Sign in" in s.cleaning.boilerplate.phrases

    def test_cleaning_manifest_path_loaded_from_outputs(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "outputs": {"manifest_path": "data/manifests/custom_cleaning_manifest.json"},
            "boilerplate": {"phrases": ["Sign in", "Log in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        s = load_settings(tmp_path)
        assert str(s.cleaning.outputs.manifest_path) == "data/manifests/custom_cleaning_manifest.json"

    def test_cleaning_step_defaults(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.cleaning.steps.normalize_unicode is True
        assert s.cleaning.steps.remove_html_artifacts is True

    def test_duplicate_line_window_size_default(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.cleaning.duplicate_lines.window_size >= 1

    def test_max_blank_lines_default(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.cleaning.blank_lines.max_blank_lines >= 0


# ---------------------------------------------------------------------------
# Structure behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeepMerge:
    def test_ingestion_input_dir_loaded(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert str(s.ingestion.input_dir) == "data/raw"

    def test_cleaning_section_loaded(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.cleaning is not None

    def test_ingestion_section_loaded(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        s = load_settings(tmp_path)
        assert s.ingestion is not None

    def test_direct_key_override_in_config_yaml(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        cfg = _base_config()
        cfg["app"] = {"name": "test-app", "environment": "test", "log_level": "WARNING"}
        _write_yaml(tmp_path / "config.yaml", cfg)
        s = load_settings(tmp_path)
        assert s.app.log_level == "WARNING"

    def test_unknown_top_level_key_fails_fast(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        cfg = _base_config()
        cfg["unexpected_section"] = {"enabled": True}
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_unknown_nested_key_fails_fast(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        cfg = _base_config()
        cfg["generation"] = {"provider": "ollama", "unknown_option": True}
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)


# ---------------------------------------------------------------------------
# Missing config file
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingConfigFile:
    def test_missing_config_yaml_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path)

    def test_missing_ingestion_section_raises_validation_error(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg.pop("ingestion", None)
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_missing_cleaning_section_raises_validation_error(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg.pop("cleaning", None)
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)


# ---------------------------------------------------------------------------
# Extension validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtensionValidation:
    def _update_extensions(self, tmp_path: Path, extensions: list[str]) -> None:
        cfg = _base_config()
        cfg["ingestion"] = {
            "input_dir": "data/raw",
            "recursive": True,
            "supported_extensions": extensions,
            "outputs": {
                "parsed_documents_path": "data/interim/parsed.jsonl",
                "cleaned_documents_path": "data/interim/cleaned.jsonl",
                "manifest_path": "data/manifests/manifest.json",
            },
        }
        _write_yaml(tmp_path / "config.yaml", cfg)

    def test_extension_without_leading_dot_fails(self, tmp_path: Path) -> None:
        self._update_extensions(tmp_path, ["txt", "md"])
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_empty_supported_extensions_fails(self, tmp_path: Path) -> None:
        self._update_extensions(tmp_path, [])
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_mixed_case_extensions_normalized(self, tmp_path: Path) -> None:
        self._update_extensions(tmp_path, [".TXT", ".HTML"])
        s = load_settings(tmp_path)
        assert ".txt" in s.ingestion.supported_extensions
        assert ".html" in s.ingestion.supported_extensions


# ---------------------------------------------------------------------------
# Cleaning mode validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCleaningModeValidation:
    def test_invalid_cleaning_mode_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {"mode": "aggressive", "boilerplate": {"phrases": ["Sign in"]}}
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)


# ---------------------------------------------------------------------------
# Duplicate-line window size validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDuplicateLineWindowSize:
    def test_window_size_zero_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "duplicate_lines": {"window_size": 0},
            "boilerplate": {"phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_window_size_negative_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "duplicate_lines": {"window_size": -3},
            "boilerplate": {"phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_window_size_one_is_valid(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "duplicate_lines": {"window_size": 1},
            "boilerplate": {"phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        s = load_settings(tmp_path)
        assert s.cleaning.duplicate_lines.window_size == 1


# ---------------------------------------------------------------------------
# Max blank lines validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMaxBlankLinesValidation:
    def test_max_blank_lines_negative_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "blank_lines": {"max_blank_lines": -1},
            "boilerplate": {"phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_max_blank_lines_zero_is_valid(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "blank_lines": {"max_blank_lines": 0},
            "boilerplate": {"phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        s = load_settings(tmp_path)
        assert s.cleaning.blank_lines.max_blank_lines == 0


# ---------------------------------------------------------------------------
# Quality threshold validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQualityThresholdValidation:
    def _cleaning_yaml_with_quality(
        self, tmp_path: Path, overcleaning: float, undercleaning: float
    ) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "boilerplate": {"phrases": ["Sign in"]},
            "quality": {
                "overcleaning_threshold": overcleaning,
                "undercleaning_threshold_for_html": undercleaning,
            },
        }
        _write_yaml(tmp_path / "config.yaml", cfg)

    def test_overcleaning_threshold_above_one_fails(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        self._cleaning_yaml_with_quality(tmp_path, 1.5, 0.95)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_overcleaning_threshold_negative_fails(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        self._cleaning_yaml_with_quality(tmp_path, -0.1, 0.95)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_undercleaning_threshold_above_one_fails(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        self._cleaning_yaml_with_quality(tmp_path, 0.10, 1.5)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_undercleaning_threshold_negative_fails(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        self._cleaning_yaml_with_quality(tmp_path, 0.10, -0.1)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_boundary_values_zero_and_one_are_valid(self, tmp_path: Path) -> None:
        _make_valid_configs(tmp_path)
        self._cleaning_yaml_with_quality(tmp_path, 0.0, 1.0)
        s = load_settings(tmp_path)
        assert s.cleaning.quality.overcleaning_threshold == 0.0
        assert s.cleaning.quality.undercleaning_threshold_for_html == 1.0


# ---------------------------------------------------------------------------
# Boilerplate max_line_length validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoilerplateMaxLineLengthValidation:
    def test_max_line_length_zero_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "boilerplate": {"max_line_length": 0, "phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)

    def test_max_line_length_negative_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "boilerplate": {"max_line_length": -10, "phrases": ["Sign in"]},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)


# ---------------------------------------------------------------------------
# Boilerplate empty phrases validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBoilerplatePhrasesValidation:
    def test_empty_phrases_list_fails(self, tmp_path: Path) -> None:
        cfg = _base_config()
        cfg["cleaning"] = {
            "mode": "conservative",
            "boilerplate": {"phrases": []},
        }
        _write_yaml(tmp_path / "config.yaml", cfg)
        with pytest.raises(ValidationError):
            load_settings(tmp_path)
