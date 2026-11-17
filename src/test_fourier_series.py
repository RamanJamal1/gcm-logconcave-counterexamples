"""Cross-check Fourier certification with independent Arb quadrature and jets."""
from flint import arb,arb_series,ctx
from certify_fourier_series import certify
from certify_arb import certify as quadrature
import json,math
from pathlib import Path

def main():
 ctx.dps=200
 rows=[]
 for m in [6,7,8]:
  series=certify(m,'1/32','1/100000','1/16',mode_cutoff=8)
  quad=quadrature({'family':'fourier_multiplier','amplitudes':['1/32','1/100000'],'frequencies':['4','8'],'phases':['0','0']},m)
  a=arb(series['signed_derivative_interval']);b=arb(quad['signed_derivative_interval'])*arb('0.0625')**m
  assert a.overlaps(b),(m,str(a),str(b))
  rows.append({'order':m,'series_interval':str(a),'independent_quadrature_interval':str(b),'overlap':True})
  print(m,'overlap passed',flush=True)
 m=48;ctx.cap=m+1;t=arb_series([0,1],prec=m+1);kappa=arb(1)/2048;L=arb(3)
 power=(-L*t/(1+kappa*t)).exp()[m]*math.factorial(m)*(-1)**m
 formula=sum((arb(math.factorial(m)//math.factorial(m-r)*math.comb(m-1,r))*kappa**r*L**(m-r) for r in range(m)),arb(0))
 assert power.overlaps(formula)
 result={'low_order_cross_checks':rows,'order48_time_change_power_series_check':True}
 Path('certs/fourier_series_independent_checks.json').write_text(json.dumps(result,indent=2));print('order48 power-series check passed')
if __name__=='__main__':main()
