"""Search convex quartic deformations of the explicit Fourier family.

The quartic adds nonnegative curvature everywhere. Numerical quadrature still
requires independent interval certification before any sign claim.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,json,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from search_windowed_modes import jets
from generated_extended_integrands import INTEGRANDS

def evaluate(a,k,lam,center,tilt,m,grid=2049):
 L=12/math.sqrt(k)+abs(tilt)/k;x=np.linspace(-L,L,grid);g,u=jets(x,a,0,m);dx=x-center
 u[2]-=k+3*lam*dx**2;u[3]-=6*lam*dx;u[4]-=6*lam
 logp=-k*x*x/2-lam*dx**4/4-tilt*x+np.log(g);density=np.exp(logp-logp.max());return float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*density,x)/np.trapezoid(density,x))

def search(m,iterations,seed):
 start=time.time();best={}
 def decode(x):
  aa=math.exp(x[0]);a=np.array([aa,aa*aa*.75**m*x[1],aa**3*(2/3)**m*x[2]]);k=np.dot(abs(a),[1,4,9])/(1-sum(abs(a)))+1e-6;return a,k,math.exp(x[3]),x[4],x[5]
 def objective(x):
  a,k,l,c,h=decode(x);value=evaluate(a,k,l,c,h,m);score=np.arcsinh(value/a[0]**2)
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,amplitudes=a.tolist(),kappa=k,quartic_lambda=l,quartic_center=c,tilt=h,signed_numerical=value,objective=float(score),certified=False,elapsed_seconds=time.time()-start);Path(f'checkpoints/convex_perturbation_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n')
  return score if np.isfinite(score) else 1e100
 r=differential_evolution(objective,[(math.log(.02),math.log(.2)),(-1,2),(-1,1),(math.log(1e-8),math.log(.03)),(-6,6),(-.1,.1)],popsize=12,maxiter=iterations,tol=1e-7,seed=1990+m+seed,polish=True)
 best.update(evaluations=r.nfev,signed_fine=evaluate(np.array(best['amplitudes']),best['kappa'],best['quartic_lambda'],best['quartic_center'],best['tilt'],m,8193));Path(f'checkpoints/convex_perturbation_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--iterations',type=int,default=180);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.iterations,a.seed)
