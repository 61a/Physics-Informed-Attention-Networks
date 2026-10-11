"""Fixed-manifest training, validation selection, and auditable test evaluation."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import CandidateDataset,read_manifest,sha256
from metrics import binary_metrics,validation_threshold
from models import Classifier,validate_input_shape


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


@torch.no_grad()
def predictions(model,loader,device):
    model.eval(); ids=[]; labels=[]; probabilities=[]
    for x,y,sample_ids in loader:
        p=model(x.to(device)).float().softmax(-1)[:,1]
        ids.extend(sample_ids); labels.extend(y.tolist()); probabilities.extend(p.cpu().tolist())
    return ids,labels,probabilities


def save_predictions(path,prediction,threshold):
    ids,labels,probabilities=prediction
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['sample_id','label','probability','predicted','threshold'])
        writer.writerows((i,y,p,int(p>=threshold),threshold) for i,y,p in zip(ids,labels,probabilities))


def train_seed(seed,cfg,rows,args):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp=bool(cfg['amp'] and device.type=='cuda')
    output=Path(args.output)/f'seed_{seed}'
    if output.exists(): raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    loaders={}
    for split in ('train','val','test'):
        dataset=CandidateDataset([r for r in rows if r['split']==split],cfg['input_shape'])
        loaders[split]=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=split=='train',num_workers=0)
    model=Classifier(cfg,args.encoder,args.front_end).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    criterion=nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'])
    history=[]; best_score=-1.; best_state=None; best_threshold=None; stale=0
    accumulation=cfg['accumulation_steps']
    for epoch in range(cfg['epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True); total_loss=0.; total_samples=0
        # Scale each accumulation group by its actual number of examples,
        # including the last partial group and last partial microbatch.
        samples=len(loaders['train'].dataset); batch=cfg['batch_size']
        for i,(x,y,_) in enumerate(loaders['train']):
            x,y=x.to(device),y.to(device)
            group_start=(i//accumulation)*accumulation*batch
            group_samples=min(accumulation*batch,samples-group_start)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                loss=criterion(model(x),y)
                scaled=loss*x.shape[0]/group_samples
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss.')
            scaler.scale(scaled).backward()
            total_loss+=loss.item()*x.shape[0]; total_samples+=x.shape[0]
            if (i+1)%accumulation==0 or i+1==len(loaders['train']):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        val=predictions(model,loaders['val'],device)
        threshold=validation_threshold(val[1],val[2]); metrics=binary_metrics(val[1],val[2],threshold)
        score=metrics['f1']
        history.append(dict(epoch=epoch+1,training_loss=total_loss/total_samples,validation=metrics))
        write_json(output/'training_log.json',history)
        print(f'seed={seed} epoch={epoch+1} validation_f1={score:.6f}',flush=True)
        if score>best_score:
            best_score=score; stale=0; best_threshold=threshold
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch=epoch+1
        else: stale+=1
        if epoch+1>=cfg['min_epochs'] and stale>=cfg['patience']: break
    model.load_state_dict(best_state)
    checkpoint=dict(state_dict=best_state,config=cfg,encoder=args.encoder,front_end=args.front_end,seed=seed,epoch=best_epoch,threshold=best_threshold)
    torch.save(checkpoint,output/'checkpoint.pt')
    val=predictions(model,loaders['val'],device)
    test=predictions(model,loaders['test'],device)
    save_predictions(output/'validation_predictions.csv',val,best_threshold)
    save_predictions(output/'test_predictions.csv',test,best_threshold)
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    provenance=dict(seed=seed,encoder=args.encoder,front_end=args.front_end,config=cfg,
                    config_sha256=sha256(args.config),
                    manifest_sha256=sha256(args.manifest),input_files=rows,source_sha256=source,
                    checkpoint_sha256=sha256(output/'checkpoint.pt'),torch_version=str(torch.__version__),
                    numpy_version=np.__version__,device=str(device),amp=amp,best_epoch=best_epoch,
                    test=binary_metrics(test[1],test[2],best_threshold))
    write_json(output/'result.json',provenance)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True); parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--encoder',default='conformer',choices=['conformer','vit_small','resnet18','efficientnet_b0','convnext_tiny','mobilenet_v3_large'])
    parser.add_argument('--front-end',default='learned',choices=['raw','uniform','learned'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,123,456,789,1024])
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    validate_input_shape(cfg)
    for key in ('batch_size','accumulation_steps','epochs','min_epochs','patience'):
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f'{key} must be a positive integer.')
    if cfg.get('gradient_checkpointing') or cfg.get('augmentation')!='none' or cfg.get('normalization')!='none':
        raise ValueError('This configuration uses no checkpointing, augmentation, or implicit normalization.')
    if cfg['min_epochs']>cfg['epochs']: raise ValueError('min_epochs exceeds epochs.')
    rows=read_manifest(args.manifest)
    for seed in args.seeds: train_seed(seed,cfg,rows,args)


if __name__=='__main__': main()
