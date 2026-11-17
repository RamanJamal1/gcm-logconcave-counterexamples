"""Batched periodic surrogate for screening; Gaussian-filter terms are omitted."""
import math,torch

def evaluate(amplitudes,kappa,order,grid=512):
 B,K=amplitudes.shape;dtype=amplitudes.dtype;device=amplitudes.device
 frequencies=torch.arange(1,K+1,dtype=dtype,device=device)
 theta=torch.arange(grid,dtype=dtype,device=device)*2*math.pi/grid
 cosine=torch.cos(frequencies[:,None]*theta[None,:])
 factors=torch.stack([(frequencies**2/2)**n/math.factorial(n) for n in range(order+1)])
 v=torch.einsum('bk,nk,kx->bnx',amplitudes,factors,cosine)
 g=1+v[:,0];logs=[torch.log1p(v[:,0])];result=torch.zeros(B,dtype=dtype,device=device)
 for n in range(1,order+1):
  logn=(v[:,n]-sum((k/n)*logs[k]*v[:,n-k] for k in range(1,n)))/g;logs.append(logn)
  h=v[:,n]*logs[0]+sum((1-k/n)*v[:,n-k]*logs[k] for k in range(1,n))
  result=result+math.comb(order-1,n-1)*kappa**(order-n)*h.mean(dim=1)
 return result*math.factorial(order)+.5*math.factorial(order-1)*kappa**order

if __name__=='__main__':
 import json
 from pathlib import Path
 from periodic_search import Periodic
 import numpy as np
 rows=[]
 for order in (12,16,17):
  amps=np.array([[1/20,1/100000],[1/32,1/262144]])
  k=np.array([1/18,1/30]);cpu=np.array([Periodic(order,2,512).evaluate(a,b) for a,b in zip(amps,k)])
  gpu=evaluate(torch.tensor(amps,device='cuda'),torch.tensor(k,device='cuda'),order).cpu().numpy();error=float(np.max(abs(gpu-cpu)/np.maximum(abs(cpu),1e-10)))
  assert error<1e-9,(order,error);rows.append({'order':order,'cpu':cpu.tolist(),'gpu':gpu.tolist(),'relative_error':error})
 Path('certs/periodic_cuda_tests.json').write_text(json.dumps({'passed':True,'checks':rows,'method':'periodic surrogate only; not the full Gaussian-windowed entropy'},indent=2)+'\n');print(rows)
