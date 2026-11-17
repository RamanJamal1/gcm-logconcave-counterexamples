"""Broader signed Fourier screening with a global Lipschitz curvature bound.

Only a periodic surrogate is optimized; Gaussian-mode corrections require
separate rigorous certification. The density's convexity bound is analytic.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import math,json,time,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from periodic_search import Periodic

def curvature(a,grid=1025):
 f=np.arange(1,len(a)+1);x=np.linspace(0,np.pi,grid);c=np.cos(f[:,None]*x);s=np.sin(f[:,None]*x);g=1+a@c;d=-(a*f)@s;dd=-(a*f*f)@c
 A=[np.dot(abs(a),f**j) for j in range(4)];den=1-A[0]
 if den<=0:return float('inf')
 lip=A[3]/den+3*A[1]*A[2]/den**2+2*A[1]**3/den**3
 return float(np.max(dd/g-(d/g)**2)+lip*np.pi/(2*(grid-1)))

def search(order,iterations):
 obj=Periodic(order,3,256);start=time.time();best={'periodic_surrogate':float('inf')}
 def evaluate(x):
  a=np.array([np.exp(x[0]),x[1]*np.exp(2*x[0]),x[2]*np.exp(3*x[0])]);k=curvature(a)*np.exp(x[3])
  if not np.isfinite(k) or k<=0:return 1e100
  value=obj.evaluate(a,k);score=np.arcsinh(value/(a[0]**2))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=order,amplitudes=a.tolist(),kappa=k,periodic_surrogate=value,objective=float(score),certified=False,note='Periodic surrogate only; full Gaussian correction is not included',elapsed_seconds=time.time()-start)
   Path(f'checkpoints/periodic_curvature_m{order}.json').write_text(json.dumps(best,indent=2)+'\n')
  return score if np.isfinite(score) else 1e100
 result=differential_evolution(evaluate,[(math.log(.003),math.log(.65)),(-2,2),(-2,2),(0,math.log(2))],popsize=24,maxiter=iterations,tol=1e-8,seed=1917+order,polish=True)
 a=np.array(best['amplitudes']);best.update(evaluations=result.nfev,periodic_surrogate_fine=Periodic(order,3,2048).evaluate(a,best['kappa']));Path(f'checkpoints/periodic_curvature_m{order}.json').write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[16,15,14,13,12]);p.add_argument('--iterations',type=int,default=300);a=p.parse_args()
 for order in a.orders:search(order,a.iterations)
