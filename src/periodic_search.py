"""Numerical periodic surrogate for multi-frequency Gaussian-windowed densities.

Nonzero Gaussian Fourier modes are omitted in this SCREENING objective.
No output of this program is a certificate or a Gaussian-entropy conclusion.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
class Periodic:
 def __init__(self,m,K,grid=512):
  self.m=m;self.K=K;self.cos=np.cos(np.arange(1,K+1)[:,None]*np.arange(grid)[None,:]*2*np.pi/grid)
  self.factors=np.array([((np.arange(1,K+1)**2/2)**n)/math.factorial(n) for n in range(m+1)])
 def evaluate(self,a,kappa):
  v=(self.factors*a[None,:])@self.cos;g=1+v[0];ell=[np.log1p(v[0])];result=0.
  for n in range(1,self.m+1):
   ln=(v[n]-sum((k/n)*ell[k]*v[n-k] for k in range(1,n)))/g;ell.append(ln)
   h=v[n]*ell[0]+sum((1-k/n)*v[n-k]*ell[k] for k in range(1,n))
   result+=math.comb(self.m-1,n-1)*kappa**(self.m-n)*h.mean()
  return result*math.factorial(self.m)+.5*math.factorial(self.m-1)*kappa**self.m
 def decode(self,x):
  a=[np.exp(x[0])]
  for j in range(2,self.K+1):
   base=sum(a[i-1]*a[j-i-1]*(1-i*(j-i)/j**2)**self.m for i in range(1,j))
   a.append(base*np.exp(x[j-1]))
  a=np.array(a);k=np.exp(x[-1])*np.dot(np.arange(1,self.K+1)**2,a)/(1-a.sum())
  return a,k
 def objective(self,x):
  a,k=self.decode(x)
  if a.sum()>=.8 or k<=0:return 1e100
  value=self.evaluate(a,k)/a[0]**2
  return np.arcsinh(value) if np.isfinite(value) else 1e100
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[16,15,14,13,12]);p.add_argument('--terms',type=int,default=3);a=p.parse_args();rows=[]
 for m in a.orders:
  obj=Periodic(m,a.terms)
  r=differential_evolution(obj.objective,[(np.log(.003),np.log(.15))]+[(np.log(.02),np.log(5))]*(a.terms-1)+[(np.log(1.0001),np.log(1.5))],popsize=20,maxiter=200,tol=1e-8,seed=4000+m)
  amps,kappa=obj.decode(r.x);value=obj.evaluate(amps,kappa)
  fine=Periodic(m,a.terms,2048).evaluate(amps,kappa)
  row={'order':m,'amplitudes':amps.tolist(),'kappa':kappa,'periodic_surrogate':value,'periodic_surrogate_fine':fine,'evaluations':r.nfev,'certified':False,'note':'Gaussian filtering modes omitted; screening only'};rows.append(row);print(row,flush=True)
  Path(f'checkpoints/periodic_{a.terms}_m{m}.json').write_text(json.dumps(row,indent=2)+'\n')
