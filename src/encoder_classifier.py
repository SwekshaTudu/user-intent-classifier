"""Frozen MiniLM sentence encoder with a trained intent classification head.

Optional CPU experiment. This module never opens the held-out test dataset and
never modifies the original classifier, annotations, or frozen evaluation.
"""
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from fetch_subtitles import request_bytes
from intent_classifier import (LABELS, QuerySignals, check_separation,
                               dataset_summary, environment, fingerprint,
                               keyword_predict, metrics)
from intent_classifier import read_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_FILE = 'onnx/model_quint8_avx2.onnx'
MODEL_SHA256 = 'b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21'
DEFAULT_ENCODER = ROOT/'models/all-MiniLM-L6-v2'/REVISION
DEFAULT_OUTPUT = ROOT/'artifacts/pretrained_encoder'


def question_stem(question):
    """Exclude enumerated event lists while retaining the requested question."""
    return re.split(r'\n\s*(?:\([a-d]\)|[①②③④])', question, maxsplit=1)[0].strip()


def mean_pool_and_normalize(tokens, mask):
    weights = np.asarray(mask, dtype=np.float32)[..., None]
    pooled = (np.asarray(tokens, dtype=np.float32)*weights).sum(axis=1)/np.maximum(weights.sum(axis=1), 1e-9)
    return pooled/np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)


def download(encoder_dir):
    """Download fixed upstream assets, with a published weight checksum."""
    encoder_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'model_id':MODEL_ID, 'revision':REVISION, 'model_file':MODEL_FILE,
                'license':'Apache-2.0', 'source':f'https://huggingface.co/{MODEL_ID}',
                'max_length':256, 'pooling':'attention-mask mean pooling followed by L2 normalization',
                'files':{}}
    for name in [MODEL_FILE, 'tokenizer.json', 'config.json', 'sentence_bert_config.json']:
        target = encoder_dir/name
        expected = MODEL_SHA256 if name == MODEL_FILE else None
        if target.exists() and (not expected or fingerprint(target)==expected):
            body = target.read_bytes()
        else:
            print(f'Downloading {name}', flush=True)
            body, _ = request_bytes(f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}')
            if expected and hashlib.sha256(body).hexdigest()!=expected:
                raise ValueError('Encoder weight checksum does not match upstream metadata')
            if name.endswith('.json'):
                json.loads(body)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.tmp')
            temporary.write_bytes(body)
            temporary.replace(target)
        manifest['files'][name] = {'sha256':hashlib.sha256(body).hexdigest(), 'bytes':len(body)}
    settings = json.loads((encoder_dir/'sentence_bert_config.json').read_text())
    if settings['max_seq_length'] != 256:
        raise ValueError('Unexpected model sequence-length configuration')
    (encoder_dir/'encoder_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Encoder ready at {encoder_dir}', flush=True)


class MiniLMEncoder:
    def __init__(self, directory, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.directory = Path(directory)
        manifest = json.loads((self.directory/'encoder_manifest.json').read_text(encoding='utf-8'))
        for name, metadata in manifest['files'].items():
            if fingerprint(self.directory/name)!=metadata['sha256']:
                raise ValueError(f'Encoder asset changed: {name}')
        self.tokenizer = Tokenizer.from_file(str(self.directory/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.directory/MODEL_FILE), sess_options=options, providers=['CPUExecutionProvider'])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, questions, batch_size=16):
        if not questions:
            return np.empty((0, 384), dtype=np.float32)
        result = []
        for offset in range(0, len(questions), batch_size):
            batch = [question_stem(q) for q in questions[offset:offset+batch_size]]
            if any(not q for q in batch):
                raise ValueError('Question must not be empty')
            encoded = self.tokenizer.encode_batch(batch)
            inputs = {'input_ids':np.asarray([r.ids for r in encoded], dtype=np.int64),
                      'attention_mask':np.asarray([r.attention_mask for r in encoded], dtype=np.int64),
                      'token_type_ids':np.asarray([r.type_ids for r in encoded], dtype=np.int64)}
            tokens = self.session.run(None, {key:value for key,value in inputs.items() if key in self.input_names})[0]
            if tokens.ndim != 3 or tokens.shape[-1] != 384:
                raise ValueError(f'Unexpected encoder output: {tokens.shape}')
            result.append(mean_pool_and_normalize(tokens, inputs['attention_mask']))
        return np.concatenate(result, axis=0)


def head_features(embeddings, questions, use_signals):
    if not use_signals:
        return embeddings
    return np.concatenate([embeddings, QuerySignals().transform(questions).toarray()], axis=1)


def train(args):
    train_rows = read_dataset(args.train)
    validation = read_dataset(args.validation)
    check_separation(train_rows, validation)
    if {r['Intent label'] for r in train_rows} != set(LABELS):
        raise ValueError('All three classes must occur in training')
    x_train = [r['Question'] for r in train_rows]
    y_train = [r['Intent label'] for r in train_rows]
    x_validation = [r['Question'] for r in validation]
    encoder = MiniLMEncoder(args.encoder_dir, args.threads)
    started = time.perf_counter()
    train_embeddings = encoder.encode(x_train)
    validation_embeddings = encoder.encode(x_validation)
    encoding_seconds = time.perf_counter()-started
    comparisons = {'keyword_baseline':metrics(validation, keyword_predict(x_validation))}
    candidates = []
    # Predeclared comparison: two regularization values, with/without the
    # same question-scope features used by the prior model. No test data.
    for signals in [False, True]:
        for c in [1.0, 10.0]:
            name = f'minilm_{"with_signals" if signals else "embeddings"}_c{c:g}'
            head = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42)
            started = time.perf_counter()
            head.fit(head_features(train_embeddings, x_train, signals), y_train)
            fit_seconds = time.perf_counter()-started
            predictions = head.predict(head_features(validation_embeddings, x_validation, signals)).tolist()
            scores = metrics(validation, predictions)
            scores['head_fit_seconds'] = fit_seconds
            comparisons[name] = scores
            candidates.append((scores['macro_f1'], name, signals, head, predictions))
    _, name, signals, head, predictions = max(candidates, key=lambda row:row[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir/'encoder_head.joblib'
    artifact = {'name':name, 'head':head, 'use_signals':signals,
                'model_id':MODEL_ID, 'revision':REVISION, 'input_fields':['Question'],
                'preprocessing':'question_stem', 'max_length':256, 'threads':args.threads,
                'encoder_manifest_sha256':fingerprint(args.encoder_dir/'encoder_manifest.json')}
    joblib.dump(artifact, model_path, compress=3)
    # Include tokenization, encoder, feature creation and classifier prediction.
    # Cache is not used in timing. All runs use the same resident CPU encoder.
    timing = []
    head.predict(head_features(encoder.encode([x_validation[0]]), [x_validation[0]], signals))
    for _ in range(3):
        for question in x_validation:
            started = time.perf_counter()
            head.predict(head_features(encoder.encode([question]), [question], signals))
            timing.append((time.perf_counter()-started)*1000)
    # Score in actual singleton production mode as well: dynamic int8 inference
    # can yield small numerical differences between singleton and batch inputs.
    singleton_predictions = [head.predict(head_features(encoder.encode([q]), [q], signals))[0] for q in x_validation]
    singleton_scores = metrics(validation, singleton_predictions)
    encoder_manifest = json.loads((args.encoder_dir/'encoder_manifest.json').read_text())
    prior_validation = None
    baseline_path = ROOT/'artifacts/training_report.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if baseline['data_sha256'] == {'train':fingerprint(args.train), 'validation':fingerprint(args.validation)}:
            selected = baseline['comparisons'][baseline['selected_model']]
            prior_validation = {'model':baseline['selected_model'], 'accuracy':selected['accuracy'], 'macro_f1':selected['macro_f1']}
    report = {'selected_model':name, 'selection_metric':'Batch validation macro-F1 among four predefined encoder heads',
              'input_fields':['Question'], 'encoder_frozen':True, 'encoder_weights_updated':False,
              'encoder':encoder_manifest, 'threads':args.threads,
              'train':dataset_summary(train_rows), 'validation':dataset_summary(validation),
              'data_sha256':{'train':fingerprint(args.train), 'validation':fingerprint(args.validation)},
              'comparisons':comparisons, 'singleton_validation':singleton_scores,
              'batch_singleton_prediction_differences':sum(a!=b for a,b in zip(predictions,singleton_predictions)),
              'prior_validation':prior_validation,
              'encoder_and_head_bytes':sum(v['bytes'] for v in encoder_manifest['files'].values())+model_path.stat().st_size,
              'head_bytes':model_path.stat().st_size, 'development_encoding_seconds':encoding_seconds,
              'latency':{'p50_ms':float(np.percentile(timing,50)), 'p95_ms':float(np.percentile(timing,95)), 'measurements':len(timing), 'scope':'Warm singleton CPU tokenization + encoder + head; excludes loading and startup.'},
              'test_used':False, 'original_test_report_modified':False,
              'annotation_warning':'AI draft intent labels; validation reused for development. This is not a new independent test result.',
              'environment':environment()}
    import onnxruntime, tokenizers
    report['environment'].update({'onnxruntime':onnxruntime.__version__, 'tokenizers':tokenizers.__version__})
    (args.output_dir/'training_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Question ID','Video ID','Question','Actual draft label','Batch prediction','Singleton prediction','Correct','Uncertain?'])
        for row, batch, single in zip(validation, predictions, singleton_predictions):
            writer.writerow([row['Question ID'],row['Video ID'],row['Question'],row['Intent label'],batch,single,single==row['Intent label'],row.get('Uncertain?', '')])
    print(json.dumps({'selected_model':name, 'validation_accuracy':singleton_scores['accuracy'], 'validation_macro_f1':singleton_scores['macro_f1'], 'prior_validation':report['prior_validation'], 'encoder_and_head_bytes':report['encoder_and_head_bytes'], 'latency':report['latency'], 'test_used':False}, indent=2))


def predict(args):
    if not args.query.strip():
        raise ValueError('Query must not be empty')
    artifact = joblib.load(args.model)
    if fingerprint(args.encoder_dir/'encoder_manifest.json') != artifact['encoder_manifest_sha256']:
        raise ValueError('Encoder assets do not match the trained head')
    encoder = MiniLMEncoder(args.encoder_dir, artifact['threads'])
    values = head_features(encoder.encode([args.query]), [args.query], artifact['use_signals'])
    label = artifact['head'].predict(values)[0]
    print(json.dumps({'question':args.query, 'intent':{'Frame':'frame','Clip':'clip','Whole-video':'whole_video'}[label]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-dir', type=Path, default=DEFAULT_ENCODER)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('download', help='Download the pinned quantized encoder assets')
    training = commands.add_parser('train', help='Train heads and evaluate on validation only')
    training.add_argument('--train', type=Path, default=ROOT/'data/model/train.csv')
    training.add_argument('--validation', type=Path, default=ROOT/'data/model/validation.csv')
    training.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    training.add_argument('--threads', type=int, default=2)
    prediction = commands.add_parser('predict', help='Route a question with encoder and saved head')
    prediction.add_argument('--model', type=Path, default=DEFAULT_OUTPUT/'encoder_head.joblib')
    prediction.add_argument('--query', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'download':download(args.encoder_dir)
        elif args.command == 'train':
            if args.threads < 1:raise ValueError('Threads must be positive')
            train(args)
        else:predict(args)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
