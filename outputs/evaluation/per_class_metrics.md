# Per-Class Metrics

Current saved evaluation metrics from `outputs/evaluation`.

## Validation

| Class | Meaning | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `0` | non-cliff | `0.9917` | `0.9755` | `0.9835` | `2202` |
| `1` | cliff | `0.8071` | `0.9262` | `0.8626` | `244` |

## Test

| Class | Meaning | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `0` | non-cliff | `0.9902` | `0.9786` | `0.9844` | `2379` |
| `1` | cliff | `0.8016` | `0.8996` | `0.8477` | `229` |

## Cliff-Class Summary

The model catches most cliff cases: validation cliff recall is `0.9262`, and test cliff recall is `0.8996`.
The test cliff precision is `0.8016`, so about 80% of predicted cliffs are true cliffs.
