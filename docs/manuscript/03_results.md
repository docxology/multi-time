# Synthetic forecast results

The demonstration validated 90 chronological daily observations with no
missing values. The holdout results below are rounded from the
`models.naive` record in `output/reports/software-demo.json`.

| Metric | Value |
| --- | ---: |
| MAE | 1.689 |
| MSE | 4.629 |
| MAPE (fraction) | 0.091 |

[@fig:holdout] shows the training history, flat forecast, and held-out
observations. The historical series contains a trend and noise, so the
individual deviations from the constant prediction vary. These results
describe one synthetic series and one forecast origin.

![Daily synthetic training history followed by a flat last-value forecast and the held-out actual values.](../../output/figures/holdout.png){#fig:holdout}
