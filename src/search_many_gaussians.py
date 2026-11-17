"""Nine-component Gaussian mixtures with an analytic global curvature bound."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,json,math
from pathlib import Path
from checkpoint_io import write_json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import simpson
from generated_extended_integrands import INTEGRANDS
ROOT=Path(__file__).resolve().parents[1]
def decode(raw):
 d=raw[0];mu=np.linspace(-d,d,9);logw=np.array(list(raw[1:][::-1])+[0]+list(raw[1:]));w=np.exp(logw-logw.max());w/=w.sum();return mu,w

def curvature_coefficient_bound(raw):
 mu,w=decode(raw);v=w*np.exp(-mu*mu/2);den=np.convolve(v,v);num=np.zeros(17)
 for i in range(9):
  for j in range(9):num[i+j]+=v[i]*v[j]*(1-(mu[i]-mu[j])**2/2)
 return float(np.min(num/den))

def evaluate(raw,m,grid=1025):
 mu,w=decode(raw);x=np.linspace(-14,14,grid);logparts=np.log(w)[:,None]+mu[:,None]*x[None,:]-mu[:,None]**2/2;maximum=logparts.max(0);parts=np.exp(logparts-maximum);g=parts.sum(0);posterior=parts/g;moments={n:np.sum(mu[:,None]**n*posterior,axis=0) for n in range(1,m+1)};cumulants={1:moments[1]}
 for n in range(2,m+1):cumulants[n]=moments[n]-sum(math.comb(n-1,k-1)*cumulants[k]*moments[n-k] for k in range(1,n))
 u={j:cumulants[j]-(1 if j==2 else 0) for j in range(2,m+1)};density=np.exp(-x*x/2+maximum)*g/math.sqrt(2*math.pi);value=float((-1)**m*simpson(INTEGRANDS[m](u)*density,x=x));variance=1+np.dot(w,mu*mu)-np.dot(w,mu)**2;return value*variance**m,value,variance

def search(m,seconds,seed,wide=False):
 start=time.time();count=0;best={};tag='wide_' if wide else '';path=ROOT/f'checkpoints/many_gaussians_{tag}m{m}_s{seed}.json'
 def obj(x):
  nonlocal count
  count+=1;curvature=curvature_coefficient_bound(x) if wide else 1-x[0]**2
  if curvature<=1e-7:return 1e3-curvature
  value,signed,var=evaluate(x,m);score=float(np.arcsinh(value/math.factorial(m-1)))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   mu,w=decode(x);best.update(order=m,raw_parameters=x.tolist(),means=mu.tolist(),weights=w.tolist(),curvature_lower_bound=curvature,normalized=value,signed=signed,variance=var,objective=score,certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print(count,value,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 r=differential_evolution(obj,([(1.,5.)] if wide else [(.3,.99999)])+[(-12,8)]*4,popsize=16,maxiter=10000,seed=174600+seed,tol=1e-12,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 fine=evaluate(np.array(best['raw_parameters']),m,8193);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,fine_normalized=fine[0],fine_signed=fine[1]);write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);p.add_argument('--wide',action='store_true');a=p.parse_args();search(a.order,a.seconds,a.seed,a.wide)
