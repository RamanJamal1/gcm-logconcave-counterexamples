"""Full Gaussian spatial search with independently variable Fourier frequencies."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,json,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from search_windowed_modes import jets
from generated_extended_integrands import INTEGRANDS

def evaluate(a,w,k,m,grid=2049):
 x=np.linspace(-12/math.sqrt(k),12/math.sqrt(k),grid);g,u=jets(x,a,0,m,w);u[2]-=k;density=np.exp(-k*x*x/2)*g;return float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*density,x)/np.trapezoid(density,x))
def search(m,iterations,seed):
 start=time.time();best={}
 def decode(x):
  aa=math.exp(x[0]);w=np.array([1,x[3],x[4]]);a=np.array([aa,aa*aa*(.5+1/w[1]**2)**m*x[1],aa**3*(.5+1.5/w[2]**2)**m*x[2]]);k=np.dot(abs(a),w*w)/(1-sum(abs(a)))+1e-6;return a,w,k
 def objective(x):
  a,w,k=decode(x)
  if sum(abs(a))>=.8 or k<=0:return 1e100
  value=evaluate(a,w,k,m);score=np.arcsinh(value/a[0]**2)
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,amplitudes=a.tolist(),frequencies=w.tolist(),kappa=k,signed_numerical=value,objective=float(score),certified=False,elapsed_seconds=time.time()-start);Path(f'checkpoints/detuned_spatial_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n')
  return score if np.isfinite(score) else 1e100
 r=differential_evolution(objective,[(math.log(.02),math.log(.2)),(-1,2),(-1,1),(1.5,2.5),(2.5,3.5)],popsize=16,maxiter=iterations,tol=1e-7,seed=2020+m+seed,x0=[math.log(.0625),.5,-.1,2,3],polish=True)
 best.update(evaluations=r.nfev,signed_fine=evaluate(np.array(best['amplitudes']),np.array(best['frequencies']),best['kappa'],m,8193));Path(f'checkpoints/detuned_spatial_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--iterations',type=int,default=180);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.iterations,a.seed)
