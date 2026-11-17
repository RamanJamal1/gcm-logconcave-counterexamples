"""Positive curvature lattice with a small strict floor; numerical screen only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
from checkpoint_io import write_json
import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from scipy.optimize import differential_evolution,brentq
from generated_extended_integrands import INTEGRANDS
ROOT=Path(__file__).resolve().parents[1]
def decode(raw):
 epsilon,T,ratio=np.exp(raw[:3]);j=np.arange(-6,7);weights=np.exp(np.where(j%3==1,raw[3],np.where(j%3==2,raw[4],0.)));return epsilon,weights,np.full(len(j),T*ratio),T*(j+raw[5])
def evaluate(raw,m,grid=1025):
 eps,w,s,c=decode(raw);ss=s[:,None]
 def potential(x):
  z=x[None,:]-c[:,None];return eps*x*x/2+np.sum(w[:,None]*(ss*np.sqrt(np.pi/2)*z*erf(z/(np.sqrt(2)*ss))+ss*ss*np.exp(-z*z/(2*ss*ss))),axis=0)
 def grad(x):return eps*x+np.sum(w*s*np.sqrt(np.pi/2)*erf((x-c)/(np.sqrt(2)*s)))
 mode=brentq(grad,float(c.min()-20),float(c.max()+20));v0=potential(np.array([mode]))[0];ends=[]
 for sign in (-1,1):
  hi=1.
  while potential(np.array([mode+sign*hi]))[0]-v0<120:hi*=2
  distance=brentq(lambda d:potential(np.array([mode+sign*d]))[0]-v0-120,0,hi);ends.append(mode+sign*distance)
 x=np.linspace(*ends,grid);extras=np.concatenate([cc+np.linspace(-8,8,max(65,grid//16))*width for cc,width in zip(c,s)]);x=np.unique(np.concatenate([x,extras[(extras>ends[0])&(extras<ends[1])]]));lp=-potential(x)+v0;density=np.exp(lp);density/=simpson(density,x=x);z=(x[None,:]-c[:,None])/ss;base=np.exp(-z*z/2);H0=np.ones_like(z);H1=z;u={2:-eps-np.sum(w[:,None]*base,axis=0)}
 for j in range(3,m+1):
  n=j-2;u[j]=-np.sum(w[:,None]*ss**(-n)*(-1)**n*H1*base,axis=0);H0,H1=H1,z*H1-n*H0
 mean=simpson(x*density,x=x);variance=simpson((x-mean)**2*density,x=x);signed=float((-1)**m*simpson(INTEGRANDS[m](u)*density,x=x));return signed*variance**m,signed,variance

def search(m,seconds,seed):
 start=time.time();count=0;best={};path=ROOT/f'checkpoints/rbf_lattice_m{m}_s{seed}.json'
 def obj(x):
  nonlocal count
  count+=1
  try:value,signed,var=evaluate(x,m);score=float(np.arcsinh(value/math.factorial(m-1)))
  except (ValueError,FloatingPointError,OverflowError):return 1e100
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   eps,w,s,c=decode(x);best.update(order=m,raw_parameters=x.tolist(),weights=w.tolist(),widths=s.tolist(),centers=c.tolist(),curvature_lower_bound=eps,normalized=value,signed=signed,variance=var,objective=score,certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print(count,value,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 r=differential_evolution(obj,[(math.log(1e-4),0),(math.log(.2),math.log(5)),(math.log(.08),math.log(.7)),(-2,2),(-2,2),(-.5,.5)],popsize=12,maxiter=10000,seed=173600+seed,tol=1e-12,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 fine=evaluate(np.array(best['raw_parameters']),m,8193);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,fine_normalized=fine[0],fine_signed=fine[1]);write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
