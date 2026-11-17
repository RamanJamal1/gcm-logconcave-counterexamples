"""Check eager versus fused CUDA evaluation, including measured numerical error."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import time,json
from pathlib import Path
import torch
from cuda_higher import evaluate
r={}
try:
 x=torch.randn(128,4,3,device='cuda',dtype=torch.float32)
 fn=lambda z:evaluate(z,'fourier_multiplier',513)
 with torch.no_grad():
  t=time.time();a=fn(x);torch.cuda.synchronize();r['eager_seconds']=time.time()-t
  fused=torch.compile(fn,mode='reduce-overhead');t=time.time();b=fused(x);torch.cuda.synchronize();r['compile_seconds']=time.time()-t
  t=time.time()
  for i in range(10):b=fused(x)
  torch.cuda.synchronize();r['fused_seconds']=(time.time()-t)/10
  r['max_relative_difference']=float(((a[0]-b[0]).abs()/a[0].abs().clamp_min(1)).max());r['status']='completed'
except Exception as e:r={'status':'error','error':repr(e)}
Path('certs/compile_benchmark.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
