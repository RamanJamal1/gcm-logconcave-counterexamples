"""Full Gaussian quadrature with free curvature slack and three Fourier phases."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,json,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from scipy.integrate import simpson
from search_windowed_modes import jets
from generated_extended_integrands import INTEGRANDS
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def decode(x,m):
 a=np.exp(x[0]);w=np.array([1,x[3],x[4]]);amps=np.array([a,a*a*(.5+1/w[1]**2)**m*x[1],a**3*(.5+1.5/w[2]**2)**m*x[2]]);bound=np.dot(abs(amps),w*w)/(1-sum(abs(amps)));return amps,w,bound*np.exp(x[5])+1e-6,np.array(x[6:])
def evaluate(a,w,k,phase,m,grid=1025):
 x=np.linspace(-12/math.sqrt(k),12/math.sqrt(k),grid);g,u=jets(x,a,0,m,w,phase);u[2]-=k;density=np.exp(-k*x*x/2)*g;return float((-1)**m*simpson(INTEGRANDS[m](u)*density,x=x)/simpson(density,x=x))
def search(m,seconds,seed):
 start=time.time();best={};count=0;path=ROOT/f'checkpoints/interior_phases_m{m}_s{seed}.json'
 def obj(x):
  nonlocal count
  count+=1;a,w,k,phase=decode(x,m)
  if sum(abs(a))>=.9 or k<=0:return 1e100
  value=evaluate(a,w,k,phase,m);score=float(np.arcsinh(value/a[0]**2))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,amplitudes=a.tolist(),frequencies=w.tolist(),phases=phase.tolist(),kappa=k,signed_numerical=value,objective=score,raw_parameters=x.tolist(),certified=False,evaluations=count,elapsed_seconds=time.time()-start,status='running');write_json(path,best);print(count,value,time.time()-start,flush=True)
  return score if np.isfinite(score) else 1e100
 r=differential_evolution(obj,[(math.log(.01),math.log(.65)),(-1,2),(-1,1),(1.5,2.5),(2.5,4),(0,math.log(5))]+[(-math.pi,math.pi)]*3,popsize=12,maxiter=10000,tol=1e-12,seed=175600+seed,polish=False,callback=lambda x,convergence:time.time()-start>=seconds)
 a,w,k,phase=decode(np.array(best['raw_parameters']),m);best.update(status='bounded_run_complete',evaluations=count,elapsed_seconds=time.time()-start,signed_fine=evaluate(a,w,k,phase,m,8193));write_json(path,best);print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--seconds',type=float,default=600);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.seconds,a.seed)
