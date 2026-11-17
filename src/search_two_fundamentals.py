"""Two interacting fundamental frequencies and resonant harmonics; numeric only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
from pathlib import Path
from checkpoint_io import write_json
import numpy as np
from scipy.optimize import differential_evolution
from search_detuned_spatial import evaluate
ROOT=Path(__file__).resolve().parents[1]
def search(m,seconds,seed):
 start=time.time();best={};count=0;path=ROOT/f'checkpoints/two_fundamentals_m{m}_s{seed}.json'
 def decode(x):
  a=np.exp(x[0]);r=x[1];c=a*r**(-m)*x[2];b=a*a*.75**m*x[3];cross=a*c*((1+r*r+(1+r)**2)/(2*(1+r)**2))**m*x[4];d=c*c*.75**m*x[5];third=a**3*(2/3)**m*x[6];amps=np.array([a,c,b,cross,d,third]);freq=np.array([1,r,2,1+r,2*r,3]);k=np.dot(abs(amps),freq**2)/(1-sum(abs(amps)))+1e-6;return amps,freq,k
 def objective(x):
  nonlocal count
  count+=1;a,w,k=decode(x)
  if sum(abs(a))>=.8 or k<=0:return 1e100
  value=evaluate(a,w,k,m,1025);score=float(np.arcsinh(value/(a[0]**2)))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,amplitudes=a.tolist(),frequencies=w.tolist(),kappa=k,signed_numerical=value,objective=score,parameters=x.tolist(),certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print('best',count,value,score,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 def callback(x,convergence):return time.time()-start>=seconds
 r=differential_evolution(objective,[(math.log(.01),math.log(.2)),(1.02,2.2),(-1,1),(-1,2),(-2,3),(-1,2),(-1,1)],popsize=12,maxiter=10000,tol=1e-8,seed=171100+seed,callback=callback,polish=False)
 a,w,k=decode(np.array(best['parameters']));best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,signed_fine=evaluate(a,w,k,m,8193));write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
