"""GPU high-order quadrature checked against independent Arb spatial integration."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,math
from pathlib import Path
import torch
from flint import arb
from cuda_higher import evaluate,parameters,FAMILIES
from certify_higher import certify
rows=[]
for family in FAMILIES:
 raw=torch.full((1,4,3),.1,device='cuda',dtype=torch.float64)
 coarse=evaluate(raw,family,2049);fine=evaluate(raw,family,4097)
 c=parameters(raw[0],family)
 r=certify(c,9,dps=100)
 value=float(arb(r['signed_derivative_interval']).mid());gpu=float(fine[1][0,0]);error=abs(value-gpu)/max(1,abs(value))
 assert error<1e-10,(family,value,gpu,error)
 convergence=float(((coarse[1]-fine[1]).abs()/fine[1].abs().clamp_min(1)).max());assert convergence<1e-10
 row={'family':family,'order9_gpu':gpu,'order9_arb_interval':r['signed_derivative_interval'],'relative_error':error,'orders9_through12_resolution_error':convergence};rows.append(row);print(row,flush=True)
Path('certs/cuda_higher_tests.json').write_text(json.dumps({'passed':True,'checks':rows},indent=2)+'\n')
