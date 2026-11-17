"""Full Gaussian-filtered two-frequency numerical search; final proof separate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,ctypes,math,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
class Evaluator:
 def __init__(self,N=40,Q=6):
  rows=[]
  for n in range(2,N+1):
   for q in range(min(Q,n)+1):
    p=n-q
    for i in range(p+1):
     for j in range(q+1):
      l=p-2*i;r=q-2*j
      if l<0 or (l==0 and r<0):continue
      weight=(-1)**n*math.comb(n,q)*math.comb(p,i)*math.comb(q,j)/(n*(n-1)*2**n)*(1 if l==0 and r==0 else 2)
      rows.append((p,q,l,r,weight))
  self.arrays=[np.ascontiguousarray([row[j] for row in rows],dtype=np.int32 if j<4 else np.float64) for j in range(5)];self.n=len(rows)
  self.lib=ctypes.CDLL('/tmp/gcm_detuned_screen.so');self.fn=self.lib.evaluate_detuned;ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');self.fn.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,ip,dp]+[ctypes.c_double]*4;self.fn.restype=ctypes.c_double
 def evaluate(self,m,a,b,k,w):return self.fn(m,self.n,*self.arrays,a,b,k,w)
 def decode(self,m,x):
  a=math.exp(x[0]);w=x[3];b=math.exp(x[1])*a*a*(.75**m);k=math.exp(x[2])*(a+w*w*b)/(1-a-b);return a,b,k,w
 def objective(self,m,x):
  a,b,k,w=self.decode(m,x);value=self.evaluate(m,a,b,k,w)/a**2
  return math.asinh(value) if math.isfinite(value) else 1e100
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[16,15,14,13,12]);a=p.parse_args();e=Evaluator()
 for m in a.orders:
  start=time.time();r=differential_evolution(lambda x:e.objective(m,x),[(math.log(.002),math.log(.2)),(math.log(.01),math.log(20)),(math.log(1.00001),math.log(1.6)),(1.6,2.7)],popsize=24,maxiter=220,tol=1e-9,seed=7100+m)
  values=e.decode(m,r.x);row={'order':m,'a':values[0],'b':values[1],'kappa':values[2],'frequency_ratio':values[3],'screening_signed_derivative':e.evaluate(m,*values),'evaluations':r.nfev,'seconds':time.time()-start,'certified':False,'method':'finite Gaussian-filtered series; numerical search without rigorous error bounds'}
  Path(f'checkpoints/detuned_m{m}.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
