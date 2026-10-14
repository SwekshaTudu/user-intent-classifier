"""Summarize validation errors; does not access test data or edit labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    with (args.artifacts/'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    errors = [r for r in rows if r['Correct'] == 'False']
    summary = {
        'questions': len(rows), 'errors': len(errors),
        'confusions': dict(Counter(f"{r['Actual draft label']} -> {r['Predicted label']}" for r in errors)),
        'uncertain_errors': sum(r['Uncertain?'] == 'Yes' for r in errors),
        'clear_draft_errors': sum(r['Uncertain?'] == 'No' for r in errors),
        'error_question_ids': [r['Question ID'] for r in errors],
        'label_warning': 'AI draft references; no labels were changed or independently verified.',
    }
    (args.artifacts/'validation_error_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    with (args.artifacts/'validation_errors.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
