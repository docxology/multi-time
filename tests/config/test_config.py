"""Tests for multi_time.config subpackage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from multi_time.config import MultiTimeConfig, load_config


class TestMultiTimeConfig:
    """Tests for MultiTimeConfig dataclass."""

    def test_default_values(self):
        config = MultiTimeConfig()
        assert config.frequency == "auto"
        assert config.forecast_horizon == 12
        assert config.confidence_level == 0.95
        assert config.imputation_strategy == "drift"
        assert isinstance(config.models, list)
        assert isinstance(config.metrics, list)

    def test_validation_passes_defaults(self):
        config = MultiTimeConfig()
        errors = config.validate()
        assert errors == []

    def test_validation_catches_bad_horizon(self):
        config = MultiTimeConfig(forecast_horizon=0)
        errors = config.validate()
        assert any("forecast_horizon" in e for e in errors)

    def test_validation_catches_bad_confidence(self):
        config = MultiTimeConfig(confidence_level=1.5)
        errors = config.validate()
        assert any("confidence_level" in e for e in errors)

    def test_validation_catches_bad_strategy(self):
        config = MultiTimeConfig(imputation_strategy="invalid")
        errors = config.validate()
        assert any("imputation_strategy" in e for e in errors)

    def test_validation_catches_bad_nlags(self):
        config = MultiTimeConfig(nlags_acf=0)
        errors = config.validate()
        assert any("nlags_acf" in e for e in errors)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_dict(self):
        config = load_config({"forecast_horizon": 24, "frequency": "D"})
        assert config.forecast_horizon == 24
        assert config.frequency == "D"

    def test_load_from_yaml(self):
        data = {"forecast_horizon": 6, "models": ["naive"]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = load_config(f.name)
        assert config.forecast_horizon == 6
        assert "naive" in config.models

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_invalid_config_raises(self):
        with pytest.raises(ValueError):
            load_config({"forecast_horizon": -1})

    def test_unknown_keys_ignored(self):
        config = load_config({"forecast_horizon": 10, "unknown_key": "ignored"})
        assert config.forecast_horizon == 10


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_setup(self):
        from multi_time.config import setup_logging
        # Should not raise
        setup_logging()

    def test_setup_with_level(self):
        from multi_time.config import setup_logging
        setup_logging(level="DEBUG")

    def test_setup_with_log_file(self):
        import tempfile
        from multi_time.config import setup_logging
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            setup_logging(level="INFO", log_file=f.name)
            assert Path(f.name).exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self):
        import logging
        from multi_time.config import get_logger
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        from multi_time.config import get_logger
        logger = get_logger("my.module.name")
        # get_logger may prepend package prefix
        assert "my.module.name" in logger.name

    def test_logger_callable(self):
        from multi_time.config import get_logger
        logger = get_logger(__name__)
        # Should be able to log without error
        logger.debug("test message from test_config.py")


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_to_dict(self):
        config = MultiTimeConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "forecast_horizon" in d
        assert "models" in d
        assert d["forecast_horizon"] == 12

