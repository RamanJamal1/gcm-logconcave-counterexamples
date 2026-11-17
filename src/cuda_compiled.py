"""Fused FP32 screening; FP64 refinement remains eager and independently checked."""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import torch
from cuda_pipeline import evaluate as eager,refine,parameters,FAMILIES
_cache={}
def evaluate(raw,family,grid=513,extent=12.):
 if raw.dtype!=torch.float32 or raw.requires_grad:return eager(raw,family,grid,extent)
 key=(family,grid,extent,tuple(raw.shape))
 if key not in _cache:_cache[key]=torch.compile(lambda z:eager(z,family,grid,extent),mode='reduce-overhead')
 return _cache[key](raw)
