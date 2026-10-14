# Implementation notes: question intent routing

## Problem and routing policy

The classifier chooses evidence scope, not the answer to a question:

- **Frame:** one relevant still image supplies the evidence, such as a sign or
  clothing colour. Locating that image may still require a search.
- **Clip:** a bounded audio/video segment supplies an action sequence, spoken
  explanation, or local context.
- **Whole-video:** a summary, exhaustive count, absence check, or comparison
  across separated segments needs broad coverage.

Audio-dependent questions can be clips or whole-video queries. Subtitles help
annotation but are not model inputs: the router may receive a question before
the relevant video is known.

## Data and evaluation

The complete supplied 10% VideoMME sample contains 90 videos and 270 questions.
The split is 162 training questions from 54 videos, 54 validation questions from
18 videos, and 54 test questions from 18 videos. Each split is balanced by video
duration, with no video or question-ID overlap. Existing question/answer task
types are not intent labels. Intent labels and rationales were AI-assisted and
remain provisional; 121 questions carry uncertainty flags. Dataset and split
provenance are in `data/model/source_manifest.json`.

Accuracy counts correct routes; macro-F1 equally weights the three classes.
Reports also contain per-class precision/recall, confusion matrices, duration
and domain slices, and CPU latency. Majority and keyword baselines establish
whether learned models add value. Training fits only training examples;
validation is reused for feature and candidate selection.

## Features and model

The first approach uses TF-IDF words/pairs plus 16 binary signals for appearance,
local actions, causal explanations, ordering, and global coverage. Logistic
regression uses balanced class weights.

The current development model uses frozen `all-MiniLM-L6-v2` weights: a pinned,
checksum-verified unsigned-int8 ONNX model. The tokenizer processes the question
stem, excluding enumerated event lists, up to 256 tokens. Attention-mask mean
pooling and L2 normalization produce a 384-dimensional embedding. A logistic
regression head is trained on these embeddings, optionally concatenated with the
same scope signals. Four predefined heads compare C=1/10, with/without signals.
Validation macro-F1 selects the C=10 head with signals. Encoder weights are not
fine-tuned, and no GPU or external inference API is required.

## Results and trade-offs

| Approach | Validation accuracy | Macro-F1 |
| --- | ---: | ---: |
| MiniLM + scope signals | 77.8% (42/54) | 0.785 |
| TF-IDF + scope signals | 66.7% (36/54) | 0.663 |
| Keyword rules | 63.0% (34/54) | 0.555 |

The encoder assets and head total 23.5 MB versus 34 KB for TF-IDF. Warm singleton
CPU latency is 6.35 ms median and 9.34 ms p95 with two encoder threads, including
tokenization and classification but excluding loading/startup. Batch and singleton
predictions agreed on validation. Model size excludes installed runtime libraries.

The earlier TF-IDF model scored 59.3% accuracy and 0.589 macro-F1 on the reserved
test, below keyword rules at 66.7% and 0.634. That result was recorded before the
encoder experiment. Because the test has been exposed, the encoder's 77.8% is
**a development validation score, not a new independent test score**. The earlier
model and test report are preserved; no labels were changed to improve scores.

## Deployment and limits

Package the ONNX encoder, tokenizer, trained head, and pinned dependencies into
a CPU worker. Load them once at startup. Each request supplies a question and
returns a route; batching amortizes encoding costs, and workers scale
independently. The repository implements a CLI; an HTTP wrapper is a proposed
deployment, not a shipped endpoint. Track latency and routing distributions, and
measure correctness on newly annotated queries.

Some questions are underspecified without answer options, and Frame/Clip scope
can depend on what the video shows versus says. More model complexity cannot
remove this ambiguity. Small samples, draft labels, and academic multiple-choice
queries limit production conclusions. No end-to-end retrieval or answer accuracy
is claimed. Fresh independently annotated data is needed for another final test.
