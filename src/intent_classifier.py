"""Lightweight question-only intent classification."""
import csv
import hashlib
import platform
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

LABELS=['Frame','Clip','Whole-video']


class QuerySignals(TransformerMixin, BaseEstimator):
    """General question cues learned by the classifier, not hard-coded routes.

    Only the question stem is inspected for these signals. Enumerated candidate
    events can be long and should not drown out the question's requested scope.
    """
    patterns = (
        r'colou?r',
        r'\b(?:wear\w*|worn|dress\w*|cloth\w*|outfit|hairstyle|hair|beard|gloves|pants|trousers)\b',
        r'\b(?:shape|icon|logo|screen|written|board|interface|diagram|photograph|picture)\b',
        r'\b(?:score|scoreboard|results|time difference)\b',
        r'\b(?:main|primary|central|overall|theme|topic|summary|summarize|title|plot|focus)\b',
        r'\b(?:order|sequence|organized|chronolog\w*)\b',
        r'\b(?:throughout|entire|whole|all|most|among|quickest|longest|different)\b',
        r'\b(?:not|never|absent|doesn.t)\b',
        r'\b(?:how many|number of|times|total)\b',
        r'\b(?:why|reason|cause\w*|purpose|explain\w*)\b',
        r'\b(?:how (?:does|do|did)|process|procedure|method|moves|action|unfold)\b',
        r'\b(?:after|before|during|change|sequence|consecutive|rally)\b',
        r'\b(?:beginning|start|opening|end|ending|last|first|second|third|middle)\b',
        r'\b(?:opinion|discuss\w*|statement|said|says|say|tell|explanation|demand|speak\w*|mentions?)\b',
        r'\b(?:what (?:is|are)|which (?:object|item)|what camera|what sport)\b',
        r'\b(?:not (?:true|correct)|which of the following statements?)\b',
    )

    def fit(self, questions, y=None):
        return self

    def transform(self, questions):
        result = []
        for question in questions:
            stem = re.split(r'\n\s*(?:\([a-d]\)|[①②③④])', question.lower(), maxsplit=1)[0]
            result.append([float(bool(re.search(pattern, stem))) for pattern in self.patterns])
        return csr_matrix(np.asarray(result, dtype=float))


def read_dataset(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,strict=True)
        required={'Question ID','Video ID','Question','Intent label'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):raise ValueError(f'Missing columns: {path}')
        rows=list(reader)
    ids=set()
    for index,row in enumerate(rows,2):
        if None in row or any(row.get(k) is None for k in required):raise ValueError(f'Malformed record: {path}:{index}')
        if not re.fullmatch(r'\d{3}-[123]',row['Question ID']) or not re.fullmatch(r'[A-Za-z0-9_-]{11}',row['Video ID']):raise ValueError(f'Invalid identity: {path}:{index}')
        if row['Question ID'] in ids or not row['Question'].strip() or row['Intent label'] not in LABELS:raise ValueError(f'Duplicate ID, empty question, or invalid label: {path}:{index}')
        ids.add(row['Question ID'])
    if not rows:raise ValueError(f'Empty dataset: {path}')
    return rows


def check_separation(train,validation):
    for key in ['Question ID','Video ID']:
        if {r[key] for r in train} & {r[key] for r in validation}:raise ValueError(f'Train and validation share {key}')


def make_model(hybrid=False,c=1.0,signals=False,signals_only=False):
    word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=12000)
    features=FeatureUnion([('word',word),('character',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=12000))]) if hybrid else word
    if signals_only:
        features = QuerySignals()
    elif signals:
        features = FeatureUnion([('word', word), ('question_signals', QuerySignals())])
    return Pipeline([('features',features),('classifier',LogisticRegression(C=c,class_weight='balanced',max_iter=1000,random_state=42))])


def keyword_predict(questions):
    result=[]
    for question in questions:
        q=question.lower()
        if re.search(r'main (?:idea|focus|topic|theme|plot)|primary focus|central theme|overall message|video (?:mainly )?about|not (?:mentioned|included)|doesn.t show up|throughout|chronological order|sequence.*(?:events|appear)',q):result.append('Whole-video')
        elif re.search(r'colou?r|wear|dress|hairstyle|shape|icon|score|written|screen|look like',q):result.append('Frame')
        else:result.append('Clip')
    return result


def metrics(rows,predictions):
    targets=[r['Intent label'] for r in rows]
    result={'questions':len(rows),'accuracy':accuracy_score(targets,predictions),'macro_f1':f1_score(targets,predictions,labels=LABELS,average='macro',zero_division=0),'per_class':classification_report(targets,predictions,labels=LABELS,output_dict=True,zero_division=0),'confusion_matrix':{'labels':LABELS,'rows_actual_columns_predicted':confusion_matrix(targets,predictions,labels=LABELS).tolist()}}
    clear=[i for i,r in enumerate(rows) if r.get('Uncertain?')=='No']
    result['clear_draft_subset']={'questions':len(clear),'accuracy':accuracy_score([targets[i] for i in clear],[predictions[i] for i in clear]) if clear else None,'note':'These are clear AI drafts, not independently reviewed ground truth.'}
    for key in ['Duration','Domain']:
        groups={}
        for value in sorted({r.get(key,'') for r in rows}-{''}):
            indexes=[i for i,r in enumerate(rows) if r.get(key)==value]
            groups[value]={'questions':len(indexes),'accuracy':accuracy_score([targets[i] for i in indexes],[predictions[i] for i in indexes])}
        result[key.lower()]=groups
    return result


def latency(model,questions,repeats=3):
    model.predict([questions[0]]);durations=[]
    for _ in range(repeats):
        for question in questions:
            started=time.perf_counter();model.predict([question]);durations.append((time.perf_counter()-started)*1000)
    return {'p50_ms':float(np.percentile(durations,50)),'p95_ms':float(np.percentile(durations,95)),'measurements':len(durations),'scope':'Warm single-query inference; excludes startup and model loading.'}


def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_summary(rows):
    return {'questions':len(rows),'videos':len({r['Video ID'] for r in rows}),'labels':dict(Counter(r['Intent label'] for r in rows)),'review_status':dict(Counter(r.get('Review status','Unspecified') for r in rows)),'uncertain':sum(r.get('Uncertain?')=='Yes' for r in rows)}


def environment():return {'python':platform.python_version(),'scikit_learn':sklearn.__version__,'numpy':np.__version__,'platform':platform.platform()}
