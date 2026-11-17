"""Independent interval proof by differentiating the exact heat density.

No reduced score polynomial or finite Fourier/log expansion is used. A local
complex-time Cauchy bound supplies a rigorous spatial tail majorant.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import math,json,time,argparse
from pathlib import Path
from fractions import Fraction
from flint import arb,acb,acb_series,ctx

def rat(s):
 f=Fraction(str(s));return arb(f.numerator)/f.denominator

def certify(source,L=120,dps=200,order_override=None):
 start=time.time();ctx.dps=dps;data=json.loads(Path(source).read_text());m=data['order'] if order_override is None else order_override;ctx.cap=m+1
 params=data['parameters'];strings=params.get('amplitudes',[params.get('a'),params.get('b')]);aa=list(map(rat,strings));k=rat(params['kappa']);K=len(aa);frequencies=list(map(rat,params.get('frequencies',list(range(1,K+1)))))
 A0=sum(map(abs,aa),arb(0))
 if not A0<1:raise ValueError('Density positivity not proven')
 curvature=k-sum((abs(a)*w*w for a,w in zip(aa,frequencies)),arb(0))/(1-A0)
 t=acb_series([0,1],prec=m+1);d=1+k*t;invd=1/d;tau=t*invd;front=d**arb('-0.5');logd=d.log();atten=[(-w*w*tau/2).exp() for w in frequencies]
 def integrand(x,analytic):
  r=1+sum((a*atten[j]*(frequencies[j]*x*invd).cos() for j,a in enumerate(aa)),acb_series([0],prec=m+1))
  if analytic and not r[0].real>0:return acb('nan')
  gaussian_exponent=-k*x*x*invd/2;density=front*gaussian_exponent.exp()*r
  entropy=density*(r.log()-logd/2+gaussian_exponent)
  return (-1)**m*math.factorial(m)*entropy[m]
 # At real x, take |t|<=rho/(1+x²). Then |Im(x/d)| <= k*rho/(2*(1-k*rho)).
 rho=arb('0.01');eps=k*rho
 if not eps<1:raise ValueError('Cauchy radius too large')
 A=sum((abs(a)*(w*w*rho/(2*(1-eps))).exp()*(abs(w)*k*rho/(2*(1-eps))).cosh() for a,w in zip(aa,frequencies)),arb(0))
 if not A<1:raise ValueError('Cauchy log analyticity not proved')
 C=(1+A)/(1-eps).sqrt();C0=-(1-eps).log()/2-(1-A).log();C2=k/(2*(1-eps));beta=k/(2*(1+eps))
 # J_n=integral_L^infinity x^n exp(-beta*x²) dx.
 LL=arb(L);exp=(-beta*LL*LL).exp();J=[arb.pi().sqrt()*(beta.sqrt()*LL).erfc()/(2*beta.sqrt()),exp/(2*beta)]
 for n in range(2,2*m+3):J.append(LL**(n-1)*exp/(2*beta)+(n-1)*J[n-2]/(2*beta))
 tail=2*math.factorial(m)*rho**(-m)*C*sum((math.comb(m,j)*(C0*J[2*j]+C2*J[2*j+2]) for j in range(m+1)),arb(0))
 z=(2*arb.pi()/k).sqrt()*(1+sum((a*(-w*w/(2*k)).exp() for a,w in zip(aa,frequencies)),arb(0)))
 total=arb(0);pieces=[];tol=arb('1e-22');print('Cauchy tail',str(tail),flush=True)
 for index in range(4*L):
  left=arb(index)/4;right=arb(index+1)/4;part=acb.integral(integrand,left,right,abs_tol=tol,rel_tol=tol,eval_limit=100000).real
  if not part.is_finite():raise ArithmeticError(f'Nonfinite integral at {index}')
  total+=2*part;pieces.append(str(part))
  if index%40==0:print('interval',index,'seconds',time.time()-start,flush=True)
 value=(total+arb(0,tail.upper()))/z
 return {'order':m,'parameters':params,'signed_derivative_interval':str(value),'certified_negative_derivative':bool(value.is_finite() and value<0),'certified':bool(value.is_finite() and value<0 and curvature>0),'convexity_lower_bound':str(curvature),'convexity_certificate':source,'normalizer_interval':str(z),'numerator_tail_bound':str(tail),'cauchy_rho':'1/100','cauchy_log_disk_radius_bound':str(A),'digits':dps,'extent':L,'intervals':pieces,'seconds':time.time()-start,'method':'exact heat-density Taylor coefficients via acb_series; rigorous complex integration; local complex-time Cauchy Gaussian-moment tail bound'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);a=p.parse_args();r=certify(a.source);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(r['certified_negative_derivative'],r['signed_derivative_interval'],flush=True)
