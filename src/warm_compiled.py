"""Warm and validate all fused low-order exploration kernels."""
import json,time
from pathlib import Path
import torch
from cuda_compiled import evaluate,eager,FAMILIES
rows=[]
for family in FAMILIES:
 x=torch.randn(2048,4,3,device='cuda',dtype=torch.float32)
 with torch.no_grad():
  a=eager(x,family);start=time.time();b=evaluate(x,family);torch.cuda.synchronize();compile_seconds=time.time()-start
  start=time.time()
  for _ in range(10):b=evaluate(x,family)
  torch.cuda.synchronize();seconds=(time.time()-start)/10
  error=float(((a[0]-b[0]).abs()/a[0].abs().clamp_min(1)).max())
  assert torch.isfinite(b[0]).all() and error<.01,(family,error)
 rows.append({'family':family,'batch':len(x),'compile_seconds':compile_seconds,'seconds':seconds,'max_relative_difference_fp32':error});print(rows[-1],flush=True)
 Path('certs/compiled_exploration.json').write_text(json.dumps({'passed_families':rows,'note':'FP32 screening only. All selected candidates receive eager FP64 refinement and resolution checks.'},indent=2)+'\n')
