"""Truncated spectral-line models: S_m^{(D')} = Gauss + sum_{d<=D'} C_d Kern_m(d/2). Shows which lines create the negative window."""
import json,sys
from spectrum import *
mp.mp.dps=40
out={}
for name,Dm in [('witness3',60),('two_mode',60),('zou32',60),('R15opt5',60)]:
    C=[mp.mpf(x) for x in json.load(open('data/spectrum_%s_D%d.json'%(name,Dm)))]; kap=CONFIGS[name]['kappa']
    print('==',name)
    for Dp in [2,4,6,8,10,12,14,16,18,20,24,30,40,60]:
        neg=[]
        for m in range(2,41):
            s=mp.factorial(m-1)*kap**m/2+sum(C[d]*Kern(m,mp.mpf(d)/2,kap) for d in range(1,Dp+1) if C[d]!=0)
            if s<0: neg.append(m)
        print('  lines d<=%2d: negative orders m<=40: %s'%(Dp,neg[:12]))
    # dominant line per m
    print('  dominant |C_d Kern_m| line d* (sign) for m=10..40 step 2:',[(m,max(range(1,61),key=lambda d:abs(C[d]*Kern(m,mp.mpf(d)/2,kap))), '+' if C[max(range(1,61),key=lambda d:abs(C[d]*Kern(m,mp.mpf(d)/2,kap)))]>0 else '-') for m in range(10,41,2)])
