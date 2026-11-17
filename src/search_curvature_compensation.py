"""Quartic curvature compensation with a jointly minimized Gaussian precision.

Unlike a positive quartic added after fixing kappa, this allows kappa to fall
below the Fourier-only convexity threshold. Numerical search, not a proof.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.integrate import simpson
from search_windowed_modes import jets
from generated_extended_integrands import INTEGRANDS
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def curvature(a,lam,grid=1025):
 x=np.linspace(-math.pi,math.pi,grid);g,u=jets(x,a,0,4);j=np.arange(1,len(a)+1);q=[np.dot(abs(a),j**r)/(1-sum(abs(a))) for r in range(1,5)];q1,q2,q3,q4=q;B4=q4+4*q1*q3+3*q2*q2+12*q1*q1*q2+6*q1**4
 # u2 is periodic; subtracting 3*lam*x^2 makes its global maximum occur
 # in [-pi,pi]. Linear-interpolation error is <= h^2 sup|u4-6lam|/8.
 return float(np.max(u[2]-3*lam*x*x)+(2*math.pi/(grid-1))**2*(B4+6*lam)/8+1e-6)
def decode(raw,m):
 a=np.exp(raw[0]);amps=np.array([a,a*a*.75**m*raw[1],a**3*(2/3)**m*raw[2]]);lam=np.exp(raw[3]);k=curvature(amps,lam);return amps,k,lam

def evaluate(a,k,lam,m,grid=1537):
 L=math.sqrt(288/(math.sqrt(k*k+288*lam)+k));x=np.linspace(-L,L,grid);g,u=jets(x,a,0,m);u[2]-=k+3*lam*x*x;u[3]-=6*lam*x;u[4]-=6*lam;density=np.exp(-k*x*x/2-lam*x**4/4)*g
 return float((-1)**m*simpson(INTEGRANDS[m](u)*density,x=x)/simpson(density,x=x))
def search(m,seconds,seed):
 start=time.time();best={};count=0;path=ROOT/f'checkpoints/curvature_compensation_m{m}_s{seed}.json'
 def obj(raw):
  nonlocal count
  count+=1;a,k,lam=decode(raw,m)
  if abs(a).sum()>=.9:return 1e100
  value=evaluate(a,k,lam,m);score=float(np.arcsinh(value/a[0]**2))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,raw_parameters=raw.tolist(),amplitudes=a.tolist(),kappa=k,quartic_lambda=lam,signed=value,objective=score,certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print(count,value,k,lam,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 differential_evolution(obj,[(math.log(.01),math.log(.35)),(-2,3),(-2,2),(math.log(1e-9),math.log(.03))],popsize=20,maxiter=10000,seed=912390+seed,tol=1e-10,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 a,k,lam=decode(np.array(best['raw_parameters']),m);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,signed_fine=evaluate(a,k,lam,m,8193));write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
