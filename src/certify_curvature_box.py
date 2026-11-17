"""Arb proof of endpoint curvature maxima on a box used in rationalization."""
import json
from pathlib import Path
from flint import arb,ctx
ctx.dps=100
def add(a,b):return [(a[j] if j<len(a) else 0)+(b[j] if j<len(b) else 0) for j in range(max(len(a),len(b)))]
def scale(a,c):return [x*c for x in a]
def mul(a,b):
 out=[arb(0)]*(len(a)+len(b)-1)
 for i,x in enumerate(a):
  for j,y in enumerate(b):out[i+j]+=x*y
 return out
def diff(a):return [j*a[j] for j in range(1,len(a))]
a=arb('0.065','0.025');b=arb('0.00005','0.00005');c=arb('-0.0000005','0.0000005');g=[1-b,a-3*c,2*b,4*c];gp=diff(g);H=add(mul(add(mul([1,0,-1],diff(gp)),scale(mul([0,1],gp),-1)),g),scale(mul([1,0,-1],mul(gp,gp)),-1));N=add(mul(diff(H),g),scale(mul(H,gp),-2));upper=N[0].upper()+sum((abs(v).upper() for v in N[1:]),arb(0));positive=1-abs(a).upper()-abs(b).upper()-abs(c).upper();assert upper<0 and positive>0
r={'certified':True,'amplitude_box':[['1/25','9/100'],['0','1/10000'],['-1/1000000','0']],'density_multiplier_lower_bound':str(positive),'N_coefficient_intervals':list(map(str,N)),'N_uniform_upper_bound':str(upper),'conclusion':'The maximum of (log g) second derivative occurs at cos(x)=-1 throughout this box. It equals (a-4b+9c)/(1-a+b-c).','digits':100};Path('certs/curvature_rationalization_box.json').write_text(json.dumps(r,indent=2)+'\n');print(r['N_uniform_upper_bound'])
