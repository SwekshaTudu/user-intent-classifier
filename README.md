# User intent classifier

Routes a text question to `frame`, `clip`, or `whole_video`. The current approach
uses a frozen MiniLM sentence encoder with a trained logistic regression head.
Prediction needs only the question. Videos, subtitles, and answer options are
annotation context, not prediction inputs.

## Run

Requirements: Python 3.12+, CPU, and internet access during setup for Python
packages and approximately 24 MB of pinned encoder assets. Prediction is local.

```bash
python -m pip install -r requirements-encoder.txt
python src/encoder_classifier.py download
python src/encoder_classifier.py train --output-dir artifacts/encoder_reproduction
python src/encoder_classifier.py predict --model artifacts/encoder_reproduction/encoder_head.joblib --query "What colour is the car?"
python -m unittest discover -s src -p "test_*.py"
```

Example output:

```json
{"question": "What colour is the car?", "intent": "frame"}
```

Run all commands from the repository root. [run.yaml](run.yaml) contains the
complete setup, execution, and test commands. It also reproduces the earlier
TF-IDF experiment and its historical holdout evaluation in separate directories.
It does not evaluate the encoder against the already exposed test set.

## Results

All reference intent labels are AI-assisted drafts. These numbers measure
agreement with those drafts, not independently verified retrieval quality.

| Approach | Validation accuracy | Validation macro-F1 | Historical test accuracy |
| --- | ---: | ---: | ---: |
| MiniLM + scope signals | 42/54 (77.8%) | 0.785 | Not independently evaluated |
| TF-IDF + scope signals | 36/54 (66.7%) | 0.663 | 32/54 (59.3%) |
| Keyword baseline | 34/54 (63.0%) | 0.555 | 36/54 (66.7%) |
| Majority baseline | 25/54 (46.3%) | 0.211 | 25/54 (46.3%) |

MiniLM's encoder assets and classification head total approximately **23.5 MB**.
Measured warm singleton CPU latency was **6.35 ms median / 9.34 ms p95**, including
tokenization, embedding, scope features, and classification. Startup, model
loading, runtime libraries, and download time are excluded. The earlier model
was about 34 KB with approximately 2 ms median warm latency. The encoder improves
development quality at a higher size and runtime cost.

## Repository guide

- [docs/README.md](docs/README.md): concise problem, method, results, limitations,
  and proposed deployment notes.
- [data/model](data/model): the canonical train/validation/test CSVs and source
  provenance. No videos or duplicate transcript tables are required to run.
- [src/README.md](src/README.md): code and optional data preparation commands.
- [artifacts/pretrained_encoder](artifacts/pretrained_encoder): recorded encoder
  head, validation predictions, and complete development report.
- [artifacts/final_evaluation](artifacts/final_evaluation): frozen **earlier
  TF-IDF model** and historical test results. These are not MiniLM test results.

Model downloads live in ignored `models/`. Reproduction outputs, inspection
images, transcript caches, and older annotation tables are excluded from Git.
Original local work has been retained rather than deleted.

## Submission

The template requires code under `src/`, an explanation under `docs/`, exact
commands in `run.yaml`, and a push to this fork's default branch before the test
deadline. No pull request is needed.

Sources: [provided VideoMME sample](https://huggingface.co/datasets/aintropy-ai/video-mme-sample10),
[upstream MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[official ONNX pooling guidance](https://sbert.net/docs/sentence_transformer/usage/efficiency.html).
