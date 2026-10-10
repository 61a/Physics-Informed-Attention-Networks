"""Complete-model CUDA forward latency measured from a trusted checkpoint."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from models import Classifier
from data import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required for this timing protocol.')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    # Only load checkpoints created by this project or another trusted source.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=Classifier(state['config'],state['encoder'],state['front_end'])
    model.load_state_dict(state['state_dict']); model.cuda().float().eval()
    x=torch.zeros((1,*state['config']['input_shape']),device='cuda',dtype=torch.float32)
    timings=[]
    with torch.inference_mode():
        for _ in range(50): model(x)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for _ in range(200):
            torch.cuda.synchronize(); start=time.perf_counter(); model(x); torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
    result=dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_parameters=sum(p.numel() for p in model.parameters()),
                mean_ms=statistics.mean(timings),sample_std_ms=statistics.stdev(timings),raw_ms=timings,
                warmup=50,repeats=200,batch_size=1,precision='FP32',gpu=torch.cuda.get_device_name(),
                torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),input_shape=list(x.shape),
                encoder=state['encoder'],front_end=state['front_end'],checkpoint_sha256=sha256(args.checkpoint),
                boundary='complete GPU-resident forward; excludes I/O, host preprocessing and host-to-device transfer')
    target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
