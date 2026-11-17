"""Localized Fourier multiplier search; screening only, not interval proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,math,time,json,argparse
from pathlib import Path
from scipy.optimize import differential_evolution
from generated_extended_integrands import INTEGRANDS

def jets(x,a,delta,m):
 frequencies=np.arange(1,len(a)+1);z=-delta*x[None,:]+1j*frequencies[:,None];base=np.exp(-delta*x[None,:]**2/2+1j*frequencies[:,None]*x[None,:]);P0=np.ones_like(z);P1=z;ds=[1+np.sum(a[:,None]*base.real,axis=0),np.sum(a[:,None]*(P1*base).real,axis=0)]
 for n in range(1,m):P0,P1=P1,z*P1-n*delta*P0;ds.append(np.sum(a[:,None]*(P1*base).real,axis=0))
 q={j:ds[j]/ds[0] for j in range(1,m+1)};u={1:q[1]}
 for n in range(2,m+1):u[n]=q[n]-sum(math.comb(n-1,j-1)*u[j]*q[n-j] for j in range(1,n))
 return ds[0],u

def curvature(a,delta,grid=2049):
 L=math.sqrt(60/delta);x=np.linspace(-L,L,grid);g,u=jets(x,a,delta,2);frequencies=np.arange(1,len(a)+1);A0=sum(abs(a));moment2=frequencies**2+delta;moment4=frequencies**4+6*frequencies**2*delta+3*delta**2
 A=[A0,np.dot(abs(a),np.sqrt(moment2)),np.dot(abs(a),moment2),np.dot(abs(a),np.sqrt(moment2*moment4)),np.dot(abs(a),moment4)];q={j:A[j]/(1-A0) for j in range(1,5)};B={1:q[1]}
 for n in range(2,5):B[n]=q[n]+sum(math.comb(n-1,j-1)*B[j]*q[n-j] for j in range(1,n))
 tail=np.dot(abs(a),delta**2*L**2+delta+frequencies**2+2*delta*L*frequencies)*np.exp(-delta*L**2/2)/(1-A0)
 return max(float(u[2].max())+(2*L/(grid-1))**2*B[4]/8,tail)+1e-6

def evaluate(a,delta,k,m,grid=2049):
 x=np.linspace(-12/math.sqrt(k),12/math.sqrt(k),grid);g,u=jets(x,a,delta,m);u[2]-=k;density=np.exp(-k*x*x/2)*g;z=np.trapezoid(density,x);return float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*density,x)/z)

def search(m,iterations,seed):
 start=time.time();best={}
 def decode(x):
  a=math.exp(x[0]);amps=np.array([a,a*a*.75**m*x[1],a**3*(2/3)**m*x[2]]);delta=math.exp(x[3]);k=curvature(amps,delta);return amps,delta,k
 def objective(x):
  a,d,k=decode(x);value=evaluate(a,d,k,m);score=np.arcsinh(value/(a[0]**2))
  if np.isfinite(score) and score<best.get('objective',float('inf')):
   best.update(order=m,amplitudes=a.tolist(),window_precision=d,kappa=k,signed_numerical=value,objective=float(score),certified=False,elapsed_seconds=time.time()-start);Path(f'checkpoints/windowed_modes_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n')
  return score if np.isfinite(score) else 1e100
 result=differential_evolution(objective,[(math.log(.01),math.log(.2)),(-1,2),(-1,1),(math.log(.003),math.log(.7))],popsize=12,maxiter=iterations,seed=811+m+seed,tol=1e-7,polish=True)
 best.update(evaluations=result.nfev,signed_fine=evaluate(np.array(best['amplitudes']),best['window_precision'],best['kappa'],m,8193));Path(f'checkpoints/windowed_modes_m{m}_s{seed}.json').write_text(json.dumps(best,indent=2)+'\n');print(best,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--iterations',type=int,default=150);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.iterations,a.seed)
