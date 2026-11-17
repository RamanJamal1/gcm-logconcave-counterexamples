"""Explore 17-component mixtures beyond coefficientwise convexity restrictions.

Sampled curvature is a search constraint, never a global certificate.
Posterior-centered moments reduce catastrophic cancellation in high cumulants.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.integrate import simpson
from generated_extended_integrands import INTEGRANDS
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def decode(raw):
 h,gamma=raw[:2];mu=np.arange(-8,9)*h;pert=np.r_[raw[2:][::-1],0,raw[2:]];lw=-gamma*mu*mu/2+pert;w=np.exp(lw-lw.max());w/=w.sum();return mu,w

def evaluate(raw,m,grid=769):
 mu,w=decode(raw);x=np.linspace(-max(abs(mu))-11,max(abs(mu))+11,grid);parts=np.log(w)[:,None]-(x[None,:]-mu[:,None])**2/2;mx=parts.max(0);parts=np.exp(parts-mx);den=parts.sum(0);post=parts/den;center=(post*mu[:,None]).sum(0);dx=mu[:,None]-center
 moments={0:np.ones_like(x),1:np.zeros_like(x)};power=dx.copy()
 for n in range(2,m+1):power*=dx;moments[n]=(post*power).sum(0)
 curvature=1-moments[2]
 if curvature.min()<=1e-6:return None,float(curvature.min()),None
 u={1:np.zeros_like(x)}
 for n in range(2,m+1):u[n]=moments[n]-sum(math.comb(n-1,k-1)*u[k]*moments[n-k] for k in range(1,n))
 u[2]-=1;density=np.exp(mx)*den/math.sqrt(2*math.pi);value=float((-1)**m*simpson(INTEGRANDS[m](u)*density,x=x));variance=1+np.dot(w,mu*mu)
 return value*variance**m,float(curvature.min()),value

def search(m,seconds,seed):
 start=time.time();best={};count=0;path=ROOT/f'checkpoints/sampled_mixture_m{m}_s{seed}.json'
 def obj(raw):
  nonlocal count
  count+=1;norm,curv,value=evaluate(raw,m)
  if norm is None:return 10-curv
  score=float(np.arcsinh(norm/math.factorial(m-1)))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   mu,w=decode(raw);best.update(order=m,raw_parameters=raw.tolist(),means=mu.tolist(),weights=w.tolist(),sampled_curvature=curv,normalized=norm,signed=value,objective=score,certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print(count,norm,curv,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 bounds=[(.1,.8),(.05,5)]+[(-2,2)]*8;x0=[.4,1]+[0]*8
 differential_evolution(obj,bounds,x0=x0,popsize=12,maxiter=10000,seed=81930+seed,tol=1e-10,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 if best:
  fine=evaluate(np.array(best['raw_parameters']),m,8193);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,fine_normalized=fine[0],fine_sampled_curvature=fine[1],fine_signed=fine[2]);write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
