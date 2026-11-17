"""Higher-order search in globally strictly convex Gaussian curvature bumps."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
import numpy as np
from scipy.special import erf
from scipy.optimize import differential_evolution
from generated_extended_integrands import INTEGRANDS
ROOT=Path(__file__).resolve().parents[1]
def decode(raw):
 w=np.exp(raw[:2]);s=np.exp(raw[2:4]);c=raw[4]
 return np.array([w[0],w[1],w[1]]),np.array([s[0],s[1],s[1]]),np.array([0,c,-c])
def evaluate(raw,m,grid=1025):
 w,s,c=decode(raw)
 def potential(x):
  z=x[None,:]-c[:,None];ss=s[:,None];return x*x/2+np.sum(w[:,None]*(ss*np.sqrt(np.pi/2)*z*erf(z/(np.sqrt(2)*ss))+ss*ss*np.exp(-z*z/(2*ss*ss))),axis=0)
 v0=potential(np.array([0]))[0];lo=0.;hi=16.
 for _ in range(25):
  mid=(lo+hi)/2
  if potential(np.array([mid]))[0]-v0<120:lo=mid
  else:hi=mid
 L=(lo+hi)/2;x=np.linspace(-L,L,grid);lp=-potential(x)+v0;density=np.exp(lp);density/=np.trapezoid(density,x);z=(x[None,:]-c[:,None])/s[:,None];base=np.exp(-z*z/2);H0=np.ones_like(z);H1=z;u={2:-1-np.sum(w[:,None]*base,axis=0)}
 for j in range(3,m+1):
  n=j-2;H=H1
  u[j]=-np.sum(w[:,None]*s[:,None]**(-n)*(-1)**n*H*base,axis=0)
  H0,H1=H1,z*H1-n*H0
 variance=np.trapezoid(x*x*density,x);signed=float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*density,x));return signed*variance**m,signed,variance

def search(m,seconds,seed):
 start=time.time();count=0;best={};path=ROOT/f'checkpoints/rbf_higher_m{m}_s{seed}.json'
 def obj(x):
  nonlocal count
  count+=1
  try:value,signed,var=evaluate(x,m);score=float(np.arcsinh(value/math.factorial(m-1)))
  except (ValueError,FloatingPointError,OverflowError):return 1e100
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,raw_parameters=x.tolist(),weights=decode(x)[0].tolist(),widths=decode(x)[1].tolist(),centers=decode(x)[2].tolist(),curvature_lower_bound=1,normalized=value,signed=signed,variance=var,objective=score,certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');path.write_text(json.dumps(best,indent=2)+'\n');print(count,value,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 result=differential_evolution(obj,[(-4,7),(-4,7),(math.log(.05),math.log(3)),(math.log(.05),math.log(3)),(0,5)],popsize=16,maxiter=10000,seed=172600+seed,tol=1e-10,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 fine=evaluate(np.array(best['raw_parameters']),m,8193);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,fine_normalized=fine[0],fine_signed=fine[1]);path.write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
