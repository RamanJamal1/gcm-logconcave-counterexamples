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

# Optional compiled FP64 loss; post-refinement values always come from eager code.
import math
_loss_cache={}
def compiled_loss(raw,family,order,grid):
 key=(family,order,grid,tuple(raw.shape))
 if key not in _loss_cache:
  def loss(z):return torch.asinh(eager(z,family,grid)[0][:,order-6]/math.factorial(order-1)).sum()
  _loss_cache[key]=torch.compile(loss,mode='reduce-overhead')
 return _loss_cache[key](raw)

def refine_fused(raw,family,order,steps=8,grid=2049):
 x=raw.detach().to(dtype=torch.float64).clone().requires_grad_(True)
 optimizer=torch.optim.LBFGS([x],max_iter=steps,lr=.3,line_search_fn='strong_wolfe')
 def closure():
  optimizer.zero_grad();loss=compiled_loss(x,family,order,grid)
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite compiled refinement')
  loss.backward();return loss
 optimizer.step(closure)
 with torch.no_grad():out=eager(x,family,grid)
 return x.detach(),tuple(v.detach() for v in out)
