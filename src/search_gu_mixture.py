"""Numerical scan of heat-regularized Gu-Sellke mixtures; not a certificate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,json,math,time
from pathlib import Path
from scipy.special import logsumexp
from generated_extended_integrands import INTEGRANDS
locations=np.array([0]+[sign*(1.8+j) for sign in [-1,1] for j in range(8)])
weights=np.array([.002]+[v for _ in range(2) for v in [.034,.093,.134,.123,.075,.031,.008,.001]])

def evaluate(T,orders,grid=4097):
 means=locations/np.sqrt(T);x=np.linspace(-max(abs(means))-12,max(abs(means))+12,grid)
 l=np.log(weights)[:,None]-(x[None,:]-means[:,None])**2/2;lp=logsumexp(l,axis=0);posterior=np.exp(l-lp);density=np.exp(lp)/np.sqrt(2*np.pi)
 moments={j:np.sum(means[:,None]**j*posterior,axis=0) for j in range(1,max(orders)+1)};u={1:moments[1]}
 for n in range(2,max(orders)+1):u[n]=moments[n]-sum(math.comb(n-1,k-1)*u[k]*moments[n-k] for k in range(1,n))
 u[2]-=1;curvature=float((-u[2]).min());values={m:float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*density,x)) for m in orders}
 return curvature,values
if __name__=='__main__':
 rows=[]
 for T in np.linspace(1,40,79):
  curvature,values=evaluate(float(T),list(range(6,18)))
  row={'noise_variance':float(T),'sampled_scaled_curvature_min':curvature,'signed_derivatives_in_scaled_coordinates':values,'certified':False};rows.append(row)
  print(row,flush=True);Path('checkpoints/gu_mixture_scan.json').write_text(json.dumps(rows,indent=2)+'\n')
