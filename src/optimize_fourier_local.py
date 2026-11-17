"""Differential-evolution numerical screening plus exact-rational Arb checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import ctypes,json,math,time,argparse
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from certify_fourier_series import gamma_numerator,certify
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]
class Evaluator:
 def __init__(self,N=48,K=12):
  rows=[]
  for n in range(2,N+1):
   for p in range(n+1):
    q=n-p
    for k in range(p%2,K+1,2):
     g=gamma_numerator(p,q,k)
     if g:rows.append((p,q,k,(-1)**n*math.comb(n,p)*g*(1 if k==0 else 2)/(n*(n-1)*2**n)))
  self.arrays=[np.ascontiguousarray([r[j] for r in rows],dtype=np.int32 if j<3 else np.float64) for j in range(4)]
  self.n=len(rows);self.lib=ctypes.CDLL('/tmp/gcm_series_screen.so');self.fn=self.lib.evaluate_series
  ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
  self.fn.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,dp,ctypes.c_double,ctypes.c_double,ctypes.c_double];self.fn.restype=ctypes.c_double
 def evaluate(self,m,a,b,k):return self.fn(m,self.n,*self.arrays,a,b,k)
 def decode(self,m,x):
  a=math.exp(x[0]);b=math.exp(x[1])*a*a*(.75**m);k=math.exp(x[2])*(a+4*b)/(1-a-b)
  return a,b,k
 def objective(self,m,x):
  try:a,b,k=self.decode(m,x);v=self.evaluate(m,a,b,k)*math.factorial(m-1)*k**m/a**2;return math.asinh(v) if math.isfinite(v) else 1e100
  except (ValueError,OverflowError):return 1e100

def main(args):
 e=Evaluator();state={'started':time.time(),'results':[],'status':'running'}
 for m in args.orders:
  result=differential_evolution(lambda x:e.objective(m,x),[(math.log(.001),math.log(.15)),(math.log(.03),math.log(30)),(math.log(1.00001),math.log(1.5))],popsize=32,maxiter=220,seed=1000+m,polish=True,tol=1e-9,workers=1)
  a,b,k=e.decode(m,result.x);row={'order':m,'evaluations':result.nfev,'parameters':[a,b,k],'screening_normalized':e.evaluate(m,a,b,k)}
  print(row,flush=True)
  if row['screening_normalized']<0:
   aa,bb,kk=[str(F(v).limit_denominator(10**10)) for v in (a,b,k)]
   r=certify(m,aa,bb,kk,N=192,dps=200,mode_cutoff=20)
   row['certification']=r
   if r['certified']:
    (ROOT/f'certs/optimized_fourier_m{m}.json').write_text(json.dumps(r,indent=2)+'\n')
    if args.sync:checkpoint(f'certified optimized Fourier violation m{m}')
    print('CERTIFIED',m,r['signed_derivative_interval'],flush=True)
  state['results'].append(row);(ROOT/'checkpoints/optimized_series_local.json').write_text(json.dumps(state,indent=2)+'\n')
 state['status']='completed';(ROOT/'checkpoints/optimized_series_local.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=list(range(9,18)));p.add_argument('--sync',action='store_true');main(p.parse_args())
