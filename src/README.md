# Code guide

- `encoder_classifier.py`: pinned encoder download, head training on train data,
  validation selection, and question-only prediction. Primary development model.
- `intent_classifier.py` and `main.py`: earlier TF-IDF classifiers, baselines,
  validation reports, and prediction.
- `evaluate.py`: frozen historical-model holdout evaluation, split checks,
  per-class metrics, and video-bootstrap accuracy intervals.
- `analyze_errors.py`: validation disagreement reports; does not edit labels.
- `fetch_subtitles.py`: optional transcript retrieval for annotation context.
- `prepare_training_data.py`: optional compact CSV preparation; an explicit
  backup is required to recover malformed annotation CSVs.

See the root README and `run.yaml` for primary execution commands. Tests use
synthetic examples for split separation, invalid input, recovery, vocabulary
isolation, pooling/padding, and feature isolation. A local encoder integration
check runs when the optional runtime and assets are present.

To download subtitle context for the canonical data, explicitly select the CSVs:

```bash
python src/fetch_subtitles.py --csv data/model/train.csv data/model/validation.csv data/model/test.csv
```

This enriches the files with timestamped transcripts and changes their checksums;
use copies if preserving recorded data fingerprints is important. It does not
change intent labels or mark them as human-reviewed. Missing files and network
failures are recorded separately. Prediction never requires subtitles.
