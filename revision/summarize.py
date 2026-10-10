"""Recompute per-run metrics from predictions before aggregating seeds."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from metrics import binary_metrics
from data import sha256


def summarize(root):
    rows=[]; signature=None; seeds=set()
    for file in sorted(Path(root).glob('seed_*/result.json')):
        record=json.loads(file.read_text(encoding='utf-8'))
        current=(record['manifest_sha256'],record['encoder'],record['front_end'],json.dumps(record['config'],sort_keys=True),json.dumps(record['source_sha256'],sort_keys=True))
        if signature is not None and current!=signature: raise ValueError('Runs use different manifests, source or model configurations.')
        signature=current
        if record['seed'] in seeds: raise ValueError('Duplicate seed.')
        seeds.add(record['seed'])
        if record['checkpoint_sha256']!=sha256(file.parent/'checkpoint.pt'): raise ValueError('Checkpoint hash mismatch.')
        with (file.parent/'test_predictions.csv').open(encoding='utf-8',newline='') as stream: predictions=list(csv.DictReader(stream))
        expected=[r for r in record['input_files'] if r['split']=='test']
        if [(r['sample_id'],int(r['label'])) for r in predictions]!=[(r['sample_id'],r['label']) for r in expected]:
            raise ValueError('Prediction identities/labels do not match the recorded test manifest.')
        threshold=record['test']['threshold']
        if any(float(r['threshold'])!=threshold for r in predictions): raise ValueError('Prediction thresholds are inconsistent.')
        metrics=binary_metrics([int(r['label']) for r in predictions],[float(r['probability']) for r in predictions],threshold)
        if metrics!=record['test']: raise ValueError(f'Metric mismatch: {file}')
        rows.append(dict(seed=record['seed'],**metrics))
    if not rows: raise ValueError('No completed seed results found.')
    summary={}
    for metric in ('precision','recall','f1','accuracy','fpr'):
        values=[r[metric] for r in rows]
        summary[metric]=dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
    return dict(n_seeds=len(rows),per_seed=rows,summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=summarize(args.runs)
    target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
