"""Check fused FP64 losses and gradients against eager code before deployment."""
import math,json,time
from pathlib import Path
import torch
from cuda_compiled import compiled_loss,eager,FAMILIES
rows=[]
for family in FAMILIES:
 for order in (7,8):
  x=(torch.randn(16,4,3,device='cuda',dtype=torch.float64)*.5).requires_grad_(True)
  old=torch.asinh(eager(x,family,2049)[0][:,order-6]/math.factorial(order-1)).sum();grad=torch.autograd.grad(old,x)[0]
  start=time.time();new=compiled_loss(x,family,order,2049);gnew=torch.autograd.grad(new,x)[0];torch.cuda.synchronize()
  error=float((grad-gnew).abs().max()/grad.abs().max().clamp_min(1));loss_error=float((old-new).abs()/old.abs().clamp_min(1));assert error<1e-9 and loss_error<1e-10,(family,order,error,loss_error)
  row={'family':family,'order':order,'compile_seconds':time.time()-start,'gradient_relative_max_error':error,'loss_relative_error':loss_error};rows.append(row);print(row,flush=True)
  Path('certs/compiled_refinement.json').write_text(json.dumps({'passed':len(rows)==8,'checks':rows},indent=2)+'\n')
