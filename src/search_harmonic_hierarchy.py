"""Signed high-harmonic periodic search with analytically enforced convexity.

This is screening, not a proof. Every Gaussian counterexample requires the
independent spatial interval certificate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,math,time,json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from periodic_search import Periodic

def search(m,K,iterations=500,seed=0):
 obj=Periodic(m,K,256);start=time.time();best={};frequencies=np.arange(1,K+1)
 def decode(x):
  a=np.exp(x[0]);amps=np.array([a]+[a**j*((j+1)/(2*j))**m*x[j-1] for j in range(2,K+1)]);k=(abs(amps)*frequencies**2).sum()/(1-abs(amps).sum())*np.exp(x[-1]);return amps,k
 def objective(x):
  amps,k=decode(x)
  if abs(amps).sum()>=.9 or k<=0:return 1e100
  value=obj.evaluate(amps,k);score=np.arcsinh(value/amps[0]**2)
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,terms=K,amplitudes=amps.tolist(),kappa=float(k),periodic_surrogate=float(value),objective=float(score),certified=False,raw=x.tolist(),elapsed_seconds=time.time()-start,note='Periodic screening only; Gaussian-filter corrections omitted')
   Path(f'checkpoints/hierarchy_K{K}_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n')
  return score if np.isfinite(score) else 1e100
 bounds=[(math.log(.015),math.log(.2))]+[(-3,3)]*(K-1)+[(1e-5,math.log(1.3))]
 x0=[math.log(.06)]+[.5 if j==2 else (-1)**j*.1 for j in range(2,K+1)]+[.001]
 r=differential_evolution(objective,bounds,popsize=24,maxiter=iterations,tol=1e-9,seed=18100+100*K+m+seed,x0=x0,polish=True)
 amps,k=decode(r.x);best.update(evaluations=r.nfev,periodic_surrogate_fine=float(Periodic(m,K,2048).evaluate(amps,k)),elapsed_seconds=time.time()-start)
 Path(f'checkpoints/hierarchy_K{K}_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
 return best
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[15,14,13,12]);p.add_argument('--terms',type=int,default=4);p.add_argument('--iterations',type=int,default=500);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 for m in a.orders:search(m,a.terms,a.iterations,a.seed)
