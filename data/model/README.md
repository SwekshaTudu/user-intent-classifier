# Canonical datasets

Use `train.csv`, `validation.csv`, and `test.csv`. Every question from the supplied
10% VideoMME sample appears exactly once: 162/54/54 questions and 54/18/18 videos.
Questions from a video stay together. Source, split policy, seed, file checksums,
and uncertainty counts are recorded in `source_manifest.json`.

`Question` is the sole model input; `Intent label` is the classification target.
IDs enforce separation. `Reason`, `Uncertain?`, and `Review status` document
annotation decisions. `Duration` and `Domain` support evaluation slices.

The intent targets are AI-assisted drafts rather than original VideoMME labels.
No subtitle or answer-choice text is passed to the classifier. Older annotation
tables and transcript caches are retained locally but excluded from submission.
`preparation_report.json` records recovery of the training table from a valid
backup after its transcript CSV formatting was damaged. The compact questions
and labels are complete and validated.
