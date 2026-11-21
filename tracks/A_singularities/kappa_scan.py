"""Curvature obstruction test: S_m(kappa) = Gauss + sum_n w_{m,n}(kappa) P_n, w>0 (mixing weights m! C(m-1,n-1) kappa^{m-n}/n!).
For fixed shape a (periodic part), vary kappa/kappa_min where kappa_min = max (log r)'' (log-concavity).  Exact mp arithmetic."""
import sys, json
from ts import *
from an2 import loadg
import numpy as np
mp.mp.dps=50
def kmin(a):
    X=np.linspace(0,2*np.pi,20001); n=np.arange(1,len(a)+1); A=np.array([float(v) for v in a])
    R=1+(A[:,None]*np.cos(np.outer(n,X))).sum(0); Rp=-(A[:,None]*n[:,None]*np.sin(np.outer(n,X))).sum(0); Rpp=-(A[:,None]*n[:,None]**2*np.cos(np.outer(n,X))).sum(0)
    return ((Rpp*R-Rp**2)/R**2).max()
res={}
for name in ['witness3','two_mode','R15opt5','zou32']:
    g=loadg(name,400,128); cfg=CONFIGS[name]; km=mp.mpf(kmin(cfg['a'])); kap0=cfg['kappa']
    print(name,'kappa_used=%.6g  kappa_min(log-concavity)=%.6g  ratio=%.4f'%(kap0,km,kap0/km))
    print('   P_n<0 for n in',[n for n in range(1,40) if g[n]<0])
    rows=[]
    for f in [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5,2,3]:
        k=km*f
        S=[None]+[S_from_h(g,k,m) if (m>=1 and k>0) else mp.factorial(m)*g[m] for m in range(1,41)]
        neg=[m for m in range(1,41) if S[m]<0]
        rows.append((f,neg[:6])); print('   kappa=%.2f*kmin: negative orders (<=40):'%f,neg[:8], ' S_15=%s'%mp.nstr(S[15],4))
    # smallest f with S_15<0 and S_14>0 etc: bisection of sign change of S_15 in f in (0,1.2)
    def S15(f): return S_from_h(g,km*f,15)
    lo,hi=mp.mpf('0.001'),mp.mpf(3)
    fs=[mp.mpf(i)/200 for i in range(1,601)]; sg=[S15(f)<0 for f in fs]
    ch=[(float(fs[i-1]),float(fs[i])) for i in range(1,len(fs)) if sg[i]!=sg[i-1]]
    print('   sign changes of S_15 as kappa/kmin varies (grid 1/200):',ch, ' S_15<0 at f in [0.005,3]?',any(sg))
    res[name]=dict(kmin=float(km),kappa_used=float(kap0),s15_sign_changes=ch)
json.dump(res,open('data/kappa_scan.json','w'),indent=1)
