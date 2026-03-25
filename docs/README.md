# 🎯 multi-time Documentation Hub

> [!NOTE]
> Comprehensive documentation for the `multi-time` package — a modular, configurable time series analysis toolkit built on sktime. Begin by reading [Notation and Glossary](technical/notation.md) to understand the unified structural definitions.

## Contents

### Core References

| Document | Description |
| --- | --- |
| [architecture.md](architecture.md) | Subpackage dependency graph, data flow, design principles |
| [api_reference.md](api_reference.md) | Full public API organized by subpackage |
| [configuration.md](configuration.md) | Config schema, available models, transforms, metrics |
| [examples.md](examples.md) | 10 end-to-end code examples |

### Module Reference (`modules/`)

Per-subpackage documentation with module maps, function tables, and test references:

| Module Doc | Subpackage |
| --- | --- |
| [modules/config.md](modules/config.md) | Configuration + logging |
| [modules/data.md](modules/data.md) | Data loading + generators |
| [modules/validate.md](modules/validate.md) | Validation + frequency detection |
| [modules/stats.md](modules/stats.md) | Descriptive + statistical tests |
| [modules/transform.md](modules/transform.md) | Transformation pipeline |
| [modules/modeling.md](modules/modeling.md) | Forecasting + probabilistic |
| [modules/evaluate.md](modules/evaluate.md) | Metrics |
| [modules/visualization.md](modules/visualization.md) | Plotting + diagnostics |
| [modules/pipeline.md](modules/pipeline.md) | End-to-end pipeline |

### Technical Reference (`technical/`)

| Document | Coverage |
| --- | --- |
| [technical/sktime_integration.md](technical/sktime_integration.md) | sktime base classes, forecasters, transformers, metrics |
| [technical/time_series_theory.md](technical/time_series_theory.md) | Stationarity, ARIMA, ETS, ACF/PACF, statistical tests |
| [technical/schema_and_api.md](technical/schema_and_api.md) | Data contracts, JSON schemas, extension points |

### Guides (`guides/`)

| Guide | Topic |
| --- | --- |
| [guides/loading_data.md](guides/loading_data.md) | CSV loading, generators, frequency handling |
| [guides/visualization_and_analysis.md](guides/visualization_and_analysis.md) | Overlays, decomposition, diagnostics, forecast plots |
| [guides/modeling_guide.md](guides/modeling_guide.md) | Baselines → ARIMA → ensemble → probabilistic |
| [guides/scripts_reference.md](guides/scripts_reference.md) | CLI scripts with full option reference |
