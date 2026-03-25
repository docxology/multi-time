"""
Configuration system for multi-time.

Provides a dataclass-based configuration with YAML/dict loading and validation.
All pipeline components read their settings from MultiTimeConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Default values ──────────────────────────────────────────────────────────────

DEFAULT_IMPUTATION_STRATEGY = "drift"
DEFAULT_FORECAST_HORIZON = 12
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MODELS = ["naive", "exp_smoothing", "auto_arima"]
DEFAULT_METRICS = ["mae", "mse", "mape"]
DEFAULT_NLAGS_ACF = 40
DEFAULT_ROLLING_WINDOW = 12
DEFAULT_SIGNIFICANCE_LEVEL = 0.05


@dataclass
class MultiTimeConfig:
    """Central configuration for all multi-time operations.

    Attributes:
        frequency: Expected time series frequency (e.g. 'D', 'H', 'M', 'auto').
        imputation_strategy: Strategy for missing-value imputation (drift, mean, ffill, bfill).
        forecast_horizon: Number of steps ahead to forecast.
        confidence_level: Confidence level for prediction intervals (0–1).
        models: List of forecaster names to run.
        metrics: List of metric names to evaluate.
        nlags_acf: Number of lags for ACF/PACF computation.
        rolling_window: Window size for rolling statistics.
        significance_level: Alpha for statistical tests.
        output_dir: Directory for saving results.
        transform_steps: Ordered list of transform names to apply.
        seasonal_period: Seasonal period for decomposition (None = auto-detect).
        log_level: Logging level string.
        log_file: Optional path to log file.
    """

    frequency: str = "auto"
    imputation_strategy: str = DEFAULT_IMPUTATION_STRATEGY
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    nlags_acf: int = DEFAULT_NLAGS_ACF
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL
    output_dir: str = "output"
    transform_steps: list[str] = field(default_factory=list)
    seasonal_period: int | None = None
    log_level: str = "INFO"
    log_file: str | None = None

    def validate(self) -> list[str]:
        """Validate configuration values, returning list of error messages."""
        errors: list[str] = []
        if self.forecast_horizon < 1:
            errors.append(f"forecast_horizon must be >= 1, got {self.forecast_horizon}")
        if not (0.0 < self.confidence_level < 1.0):
            errors.append(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if not (0.0 < self.significance_level < 1.0):
            errors.append(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
        if self.nlags_acf < 1:
            errors.append(f"nlags_acf must be >= 1, got {self.nlags_acf}")
        if self.rolling_window < 2:
            errors.append(f"rolling_window must be >= 2, got {self.rolling_window}")

        valid_strategies = {"drift", "mean", "median", "ffill", "bfill", "nearest"}
        if self.imputation_strategy not in valid_strategies:
            errors.append(
                f"imputation_strategy must be one of {valid_strategies}, "
                f"got '{self.imputation_strategy}'"
            )

        if errors:
            for err in errors:
                logger.error("Config validation error: %s", err)
        else:
            logger.info("Configuration validated successfully")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain dictionary.

        Returns:
            Dictionary of all config fields with current values.
        """
        from dataclasses import asdict
        return asdict(self)


def load_config(source: str | Path | dict[str, Any]) -> MultiTimeConfig:
    """Load configuration from a YAML file path or a dictionary.

    Args:
        source: Path to a YAML file, or a dict of config values.

    Returns:
        Validated MultiTimeConfig instance.

    Raises:
        FileNotFoundError: If YAML path does not exist.
        ValueError: If configuration validation fails.
    """
    if isinstance(source, dict):
        data = source
        logger.info("Loading config from dict")
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", path)

    config = MultiTimeConfig(**{k: v for k, v in data.items() if hasattr(MultiTimeConfig, k)})
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

    return config
