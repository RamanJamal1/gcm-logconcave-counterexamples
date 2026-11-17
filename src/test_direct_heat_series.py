"""Independent numerical check: differentiate exact heat density before quadrature.

Arb Taylor arithmetic is used at quadrature nodes, but the GH nodes and weights
are double precision. This is deliberately a numerical check, not a certificate.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,time
from pathlib import Path
from scipy.special import roots_hermitenorm
from flint import arb,arb_series,ctx
from certify_fourier_series import rat

def check(path,nodes):
 c=json.loads(Path(path).read_text());m=c['order'];ctx.dps=100;ctx.cap=m+1
 a,b,k=map(rat,[c['parameters'][s] for s in ['a','b','kappa']])
 t=arb_series([0,1],prec=m+1);d=1+k*t;tau=t/d;root=(k*d).sqrt()
 ea=(-tau/2).exp();eb=(-2*tau).exp();ld=d.log()/2
 total=arb(0);x,w=roots_hermitenorm(nodes)
 for z,weight in zip(x,w):
  if weight==0:continue
  y=arb(str(z))/root
  r=1+a*ea*y.cos()+b*eb*(2*y).cos()
  # Terms constant in time from log Z are harmless since integral r is constant.
  h=r*(r.log()-ld-arb(str(z))**2/2)
  total+=arb(str(weight))*h[m]
 xi=1+a*(-1/(2*k)).exp()+b*(-2/k).exp()
 value=total/((2*arb.pi()).sqrt()*xi)*math.factorial(m)*(-1)**m
 return {'nodes':nodes,'signed_derivative':str(value),'midpoint':float(value.mid())}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('certificate');p.add_argument('--nodes',nargs='+',type=int,default=[256,512,1024,2048]);a=p.parse_args();rows=[]
 for n in a.nodes:
  r=check(a.certificate,n);rows.append(r);print(r,flush=True)
 Path('certs/direct_heat_series_check.json').write_text(json.dumps({'method':'GH quadrature of independently differentiated exact heat density; numerical only','certificate':a.certificate,'results':rows},indent=2)+'\n')
