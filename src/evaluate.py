"""Evaluate a frozen selected model on held-out questions, without fitting."""
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from intent_classifier import (check_separation, dataset_summary, environment,
                               fingerprint, keyword_predict, latency, metrics,
                               read_dataset)


def validate_holdout(train, validation, test):
    """Reject shared videos or question IDs between any pair of splits."""
    check_separation(train, validation)
    check_separation(train, test)
    check_separation(validation, test)


def video_bootstrap_accuracy(rows, predictions, draws=2000):
    """Resample videos, retaining their question groups, for a descriptive CI."""
    groups = {}
    for row, prediction in zip(rows, predictions):
        groups.setdefault(row['Video ID'], []).append(int(row['Intent label'] == prediction))
    groups = list(groups.values())
    rng = np.random.default_rng(42)
    results = []
    for _ in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        observations = [correct for i in sampled for correct in groups[i]]
        results.append(float(np.mean(observations)))
    lower, upper = np.percentile(results, [2.5, 97.5])
    return {'lower':float(lower), 'upper':float(upper), 'confidence':0.95,
            'bootstrap_draws':draws, 'unit':'video',
            'note':'Descriptive uncertainty from video resampling; excludes annotation error and dataset shift.'}


def final_evaluate(args):
    model_path = args.model.resolve()
    training_report_path = args.training_report.resolve()
    training_report = json.loads(training_report_path.read_text(encoding='utf-8'))
    # This verifies that the evaluated model uses the same source data as selection.
    for name, path in [('train', args.train), ('validation', args.validation)]:
        if fingerprint(path) != training_report['data_sha256'][name]:
            raise ValueError(f'{name} data differs from the data used for model selection')
    state = {
        'model_sha256':fingerprint(model_path),
        'classifier_source_sha256':fingerprint(Path(__file__).with_name('intent_classifier.py')),
        'evaluator_source_sha256':fingerprint(Path(__file__)),
        'train_sha256':fingerprint(args.train),
        'validation_sha256':fingerprint(args.validation),
        'test_sha256':fingerprint(args.test),
        'selected_model':training_report['selected_model'],
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output/'freeze.json'
    report_path = output/'test_report.json'
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding='utf-8'))
        if previous['fingerprints'] != state:
            raise ValueError('Frozen evaluation inputs changed. Preserve the existing report; do not tune on test results.')
        if fingerprint(output/'frozen_model.joblib') != state['model_sha256']:
            raise ValueError('The frozen model snapshot has changed')
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            print('Existing frozen evaluation; returning saved scores without scoring again.')
            print(json.dumps(report['summary'], indent=2))
            return
    else:
        shutil.copy2(model_path, output/'frozen_model.joblib')
        shutil.copy2(training_report_path, output/'model_selection_report.json')
        freeze_path.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(), 'fingerprints':state}, indent=2)+'\n', encoding='utf-8')

    # No fitting, feature changes, candidate selection, or label edits below.
    train = read_dataset(args.train)
    validation = read_dataset(args.validation)
    test = read_dataset(args.test)
    validate_holdout(train, validation, test)
    artifact = joblib.load(output/'frozen_model.joblib')
    if artifact['name'] != state['selected_model'] or artifact['input_fields'] != ['Question']:
        raise ValueError('Model artifact does not match the selected question-only model')
    questions = [r['Question'] for r in test]
    majority = Counter(r['Intent label'] for r in train).most_common(1)[0][0]
    predictions = {
        'selected_model':artifact['model'].predict(questions).tolist(),
        'majority_baseline':[majority]*len(test),
        'keyword_baseline':keyword_predict(questions),
    }
    comparisons = {}
    for name, values in predictions.items():
        comparisons[name] = metrics(test, values)
        comparisons[name]['accuracy_video_bootstrap_95_ci'] = video_bootstrap_accuracy(test, values)
    fields = ['Question ID','Video ID','Question','Reference draft label',
              'Model prediction','Majority prediction','Keyword prediction',
              'Model correct','Uncertain?']
    with (output/'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, row in enumerate(test):
            writer.writerow([row['Question ID'],row['Video ID'],row['Question'],row['Intent label'],
                             predictions['selected_model'][i],predictions['majority_baseline'][i],
                             predictions['keyword_baseline'][i],
                             predictions['selected_model'][i]==row['Intent label'],row.get('Uncertain?', '')])
    normalize = lambda question: ' '.join(question.lower().split())
    training_wordings = {normalize(r['Question']) for r in train+validation}
    report = {
        'summary':{name:{'accuracy':result['accuracy'], 'macro_f1':result['macro_f1']} for name,result in comparisons.items()},
        'selected_model':state['selected_model'], 'input_fields':['Question'],
        'test':dataset_summary(test), 'comparisons':comparisons,
        'model_bytes':model_path.stat().st_size,
        'latency':latency(artifact['model'], questions), 'environment':environment(),
        'fingerprints':state,
        'shared_question_wordings_with_development':sorted({normalize(q) for q in questions} & training_wordings),
        'annotation_warning':'All test labels are AI-assisted drafts; scores measure agreement with those labels, not independently established routing or answer correctness.',
        'protocol':'Model and feature code frozen before scoring. Baselines fixed during development. No fitting or model selection on test. Repeated unchanged runs return saved results.',
        'limitations':['Small holdout: 18 videos and 54 questions.',
                       'Video-MME is multiple-choice video QA, not a production query distribution.',
                       'Some draft labels depend on answer options or assumptions about visual evidence, while model input is question text alone.',
                       'End-to-end retrieval and answer accuracy are not measured.'],
    }
    if fingerprint(model_path) != state['model_sha256'] or fingerprint(args.test) != state['test_sha256']:
        raise ValueError('Model or test data changed during evaluation')
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))

