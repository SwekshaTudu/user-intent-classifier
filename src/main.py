"""Train and use a CPU-based question-only intent classifier."""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier

from intent_classifier import LABELS,check_separation,dataset_summary,environment,fingerprint,keyword_predict,latency,make_model,metrics,read_dataset


def train(args):
    training=read_dataset(args.train);validation=read_dataset(args.validation)
    check_separation(training,validation)
    if {r['Intent label'] for r in training}!=set(LABELS):raise ValueError('Training data must contain all three classes')
    x_train=[r['Question'] for r in training];y_train=[r['Intent label'] for r in training]
    x_validation=[r['Question'] for r in validation]
    majority=DummyClassifier(strategy='most_frequent').fit(x_train,y_train)
    comparisons={'majority_baseline':metrics(validation,majority.predict(x_validation)),'keyword_baseline':metrics(validation,keyword_predict(x_validation))}
    candidates=[]
    # Initial three candidates plus two feature-based candidates added after
    # validation error review. Validation is development data, not final testing.
    for name,hybrid,c,signals,signals_only in [('word_logistic_c1',False,1.0,False,False),('word_logistic_c4',False,4.0,False,False),('word_char_logistic_c1',True,1.0,False,False),('signals_logistic_c1',False,1.0,False,True),('word_signals_logistic_c1',False,1.0,True,False)]:
        model=make_model(hybrid=hybrid,c=c,signals=signals,signals_only=signals_only);started=time.perf_counter()
        model.fit(x_train,y_train);fit_seconds=time.perf_counter()-started
        predictions=model.predict(x_validation).tolist();result=metrics(validation,predictions)
        result['fit_seconds']=fit_seconds;comparisons[name]=result
        candidates.append((result['macro_f1'],name,model,predictions))
    _,name,model,predictions=max(candidates,key=lambda item:item[0])
    args.output_dir.mkdir(parents=True,exist_ok=True);model_path=args.output_dir/'intent_model.joblib'
    joblib.dump({'model':model,'name':name,'input_fields':['Question'],'label_status':'AI-assisted drafts; human review pending','environment':environment()},model_path,compress=3)
    with (args.output_dir/'validation_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['Question ID','Video ID','Question','Actual draft label','Predicted label','Correct','Uncertain?'])
        for row,prediction in zip(validation,predictions):writer.writerow([row['Question ID'],row['Video ID'],row['Question'],row['Intent label'],prediction,prediction==row['Intent label'],row.get('Uncertain?','')])
    normalize=lambda q:' '.join(q.lower().split())
    report={'selected_model':name,'selection_metric':'Validation macro-F1 across five ML candidates; baselines reported separately.','development_history':'Three initial text models, followed by two question-signal models after validation error inspection. Validation was reused for feature development; it is not an unbiased final estimate.','input_fields':['Question'],'annotation_warning':'Scores measure agreement with provisional AI annotations, not independently verified routing correctness. Subtitles and options are excluded from model inputs.','test_used':False,'train':dataset_summary(training),'validation':dataset_summary(validation),'data_sha256':{'train':fingerprint(args.train),'validation':fingerprint(args.validation)},'shared_question_wordings':sorted({normalize(q) for q in x_train}&{normalize(q) for q in x_validation}),'comparisons':comparisons,'selected_training_metrics':metrics(training,model.predict(x_train).tolist()),'model_bytes':model_path.stat().st_size,'latency':latency(model,x_validation),'environment':environment()}
    (args.output_dir/'training_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'selected_model':name,'validation_accuracy':comparisons[name]['accuracy'],'validation_macro_f1':comparisons[name]['macro_f1'],'majority_accuracy':comparisons['majority_baseline']['accuracy'],'keyword_macro_f1':comparisons['keyword_baseline']['macro_f1'],'model_bytes':report['model_bytes'],'latency':report['latency'],'annotation_warning':report['annotation_warning'],'test_used':False},indent=2))


def predict(args):
    if not args.query.strip():raise ValueError('Query must not be empty')
    artifact=joblib.load(args.model)
    label=artifact['model'].predict([args.query])[0]
    print(json.dumps({'question':args.query,'intent':{'Frame':'frame','Clip':'clip','Whole-video':'whole_video'}[label]},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    training=commands.add_parser('train',help='Fit on training data and compare on validation only')
    training.add_argument('--train',type=Path,default=Path('data/model/train.csv'))
    training.add_argument('--validation',type=Path,default=Path('data/model/validation.csv'))
    training.add_argument('--output-dir',type=Path,default=Path('artifacts'));training.set_defaults(action=train)
    prediction=commands.add_parser('predict',help='Route one question using the saved model')
    prediction.add_argument('--query',required=True)
    prediction.add_argument('--model',type=Path,default=Path('artifacts/intent_model.joblib'));prediction.set_defaults(action=predict)
    evaluation=commands.add_parser('evaluate',help='Evaluate the frozen model on held-out test data; never fit')
    evaluation.add_argument('--model',type=Path,default=Path('artifacts/intent_model.joblib'))
    evaluation.add_argument('--training-report',type=Path,default=Path('artifacts/training_report.json'))
    evaluation.add_argument('--train',type=Path,default=Path('data/model/train.csv'))
    evaluation.add_argument('--validation',type=Path,default=Path('data/model/validation.csv'))
    evaluation.add_argument('--test',type=Path,default=Path('data/model/test.csv'))
    evaluation.add_argument('--output-dir',type=Path,default=Path('artifacts/final_evaluation'))
    from evaluate import final_evaluate
    evaluation.set_defaults(action=final_evaluate)
    args=parser.parse_args()
    try:args.action(args)
    except (ValueError,OSError,csv.Error) as exc:parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':main()
