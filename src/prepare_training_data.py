"""Prepare question-only model data; retain original annotation CSVs."""
import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = ['Question ID','Video ID','Question','Intent label','Reason','Uncertain?','Review status','Duration','Domain']
LABELS = {'Frame','Clip','Whole-video'}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,strict=True))


def valid(rows):
    return bool(rows) and all(None not in r and re.fullmatch(r'\d{3}-[123]',r.get('Question ID','')) and r.get('Intent label') in LABELS and r.get('Question','').strip() for r in rows) and len({r['Question ID'] for r in rows})==len(rows)


def recover(source,backup):
    rows=read(backup)
    if not valid(rows): raise ValueError('Invalid backup')
    blocks=re.split(r'(?m)^(?=\d{3}-[123],)',source.read_text(encoding='utf-8-sig'))[1:]
    if len(blocks)!=len(rows): raise ValueError('Backup and source record counts differ')
    for block,row in zip(blocks,rows):
        match=re.search(r',"?(Frame|Clip|Whole-video)"?,',block)
        if not block.startswith(f"{row['Question ID']},{row['Video ID']},") or not match or match.group(1)!=row['Intent label']:
            raise ValueError(f"Edited or unidentifiable record: {row['Question ID']}")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recover-train-from',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('data/model'))
    args=parser.parse_args()
    prepared={};report={}
    for name in ['train','validation']:
        path=Path('data/splits')/f'{name}.csv';source=path
        try:
            rows=read(path)
            if not valid(rows):raise ValueError('Malformed records')
        except (csv.Error,ValueError):
            if name!='train' or not args.recover_train_from:raise ValueError(f'{path} is malformed; supply an explicit backup')
            rows=recover(path,args.recover_train_from);source=args.recover_train_from
        prepared[name]=rows
        report[name]={'questions':len(rows),'annotation_source':str(source),'recovered':source!=path}
    for field in ['Question ID','Video ID']:
        if {r[field] for r in prepared['train']} & {r[field] for r in prepared['validation']}:raise ValueError(f'Overlap in {field}')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in prepared.items():
        with (args.output_dir/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (args.output_dir/'preparation_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
