"""Validated manifests and candidate-window loading."""
import csv
import hashlib
import os
from pathlib import Path
import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    manifest=Path(path).resolve()
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'path','label','split'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires path,label,split columns.')
        rows=list(reader)
    if not rows: raise ValueError('Manifest is empty.')
    seen=set(); groups={}; counts={s:[0,0] for s in ('train','val','test')}
    result=[]
    for row in rows:
        p=Path(row['path'])
        p=(manifest.parent/p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        key=os.path.normcase(str(p))
        if key in seen: raise ValueError(f'Duplicate sample: {p}')
        seen.add(key)
        if row['label'] not in {'0','1'} or row['split'] not in counts:
            raise ValueError('Labels must be 0/1; splits must be train/val/test.')
        group=(row.get('group') or '').strip()
        if group:
            if group in groups and groups[group]!=row['split']:
                raise ValueError(f'Observation group crosses splits: {group}')
            groups[group]=row['split']
        label=int(row['label']); counts[row['split']][label]+=1
        result.append(dict(path=str(p),sample_id=row['path'],label=label,split=row['split'],group=group,sha256=sha256(p)))
    if any(min(c)==0 for c in counts.values()):
        raise ValueError('Every split must contain both classes.')
    # Different filenames can contain duplicate arrays; reject identical files.
    hashes={}
    for row in result:
        old=hashes.get(row['sha256'])
        if old is not None: raise ValueError(f'Identical input files: {old} and {row["sample_id"]}')
        hashes[row['sha256']]=row['sample_id']
    return result


class CandidateDataset:
    def __init__(self,rows,shape): self.rows=rows; self.shape=tuple(shape)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        import torch
        row=self.rows[index]
        x=np.load(row['path'],allow_pickle=False)
        if x.shape!=self.shape or not np.issubdtype(x.dtype,np.number) or np.iscomplexobj(x) or not np.isfinite(x).all():
            raise ValueError(f'Invalid candidate array: {row["sample_id"]}; expected finite real {self.shape}.')
        x=x.astype(np.float32)
        if not np.isfinite(x).all(): raise ValueError('Input is outside FP32 range.')
        return torch.from_numpy(x),row['label'],row['sample_id']
