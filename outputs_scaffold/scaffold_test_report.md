# Scaffold Split Generalization Test

This run tests whether the high metrics survive a stricter split where molecule scaffolds do not overlap across train, validation, and test.

## Split Summary

Source data: union of `data/train_pairs.csv`, `data/val_pairs.csv`, and `data/test_pairs.csv`.

| Split | Pairs | Molecules | Scaffolds | Non-Cliff | Cliff | Cliff Rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | `41760` | `28051` | `7398` | `37407` | `4353` | `0.1042` |
| Val | `2411` | `2352` | `344` | `2023` | `388` | `0.1609` |
| Test | `3190` | `2468` | `339` | `2734` | `456` | `0.1429` |
| Dropped cross-split pairs | `13183` | `13752` | `5751` | `12296` | `887` | `0.0673` |

No scaffold overlap was allowed across train, validation, and test. Pairs whose two molecules landed in different scaffold splits were dropped.

## Metrics

| Split | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Train | `0.9429` | `0.9907` | `0.8594` | `0.8038` | `0.9233` | `0.625` |
| Val | `0.9617` | `0.9920` | `0.8931` | `0.8722` | `0.9149` | `0.625` |
| Test | `0.9622` | `0.9929` | `0.8823` | `0.8542` | `0.9123` | `0.625` |

## Test Per-Class Metrics

| Class | Meaning | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| `0` | non-cliff | `0.9852` | `0.9740` | `0.9796` | `2734` |
| `1` | cliff | `0.8542` | `0.9123` | `0.8823` | `456` |

## Conclusion

The high metrics do not disappear under the scaffold split. The test PR-AUC is `0.9622`, and the cliff-class F1 is `0.8823`.

This does not show evidence of overfitting. In fact, the scaffold test performed slightly better than the previous split. One caveat is that the scaffold split changed class balance: validation and test have higher cliff rates than train, so this is not a perfectly distribution-matched holdout.

For an even stricter check, use an external dataset or a temporal/source-based holdout generated independently from the training pairs.
