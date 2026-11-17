"""Measure actual simultaneous 65,536-candidate FP32 GPU evaluation."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
import time,json,resource
from pathlib import Path
import torch
from cuda_compiled import evaluate
rows=[]
for family in ['fourier_multiplier','rbf_curvature']:
 torch.cuda.reset_peak_memory_stats();x=torch.randn(65536,4,3,device='cuda',dtype=torch.float32);start=time.time()
 with torch.no_grad():y=evaluate(x,family,513)
 torch.cuda.synchronize();row={'family':family,'simultaneous_candidates':len(x),'finite':bool(torch.isfinite(y[0]).all()),'seconds_including_compilation':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'process_peak_host_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};assert row['finite'];rows.append(row);print(row,flush=True)
 Path('certs/large_cuda_batch.json').write_text(json.dumps({'passed_families':rows},indent=2)+'\n')
 del x,y;torch.cuda.empty_cache()
