"""For fixed a1 (kappa = (1+margin)*max (log r)''), random higher modes (K=3..5): minimum FIRST-negative order seen (m<=30). Float64 screen; surrogate only."""
import numpy as np, sys, json, time
from scan import S_all, logconc
a1s=[float(v) for v in sys.argv[2:]]; N=int(sys.argv[1]); out={}
for a1 in a1s:
    rng=np.random.default_rng(int(a1*1e6)); best={}; t0=time.time()
    for it in range(N):
        K=rng.integers(2,5)
        a=[a1]+[rng.choice([-1,1])*10**rng.uniform(-3.2*k-0.5,-0.8*k) for k in range(1,K)]
        lc=logconc(a)
        if lc<=0: continue
        kap=lc*(1+rng.choice([0,0,0.01,0.05]))
        S=S_all(a,kap,28)
        if S is None: continue
        neg=[m for m in range(2,29) if S[m]<0]
        if not neg: continue
        f=neg[0]
        if f not in best: best[f]=(a,kap)
    mn=min(best) if best else None
    out[a1]=dict(min_first_neg=mn,example=best.get(mn),hist_orders=sorted(best))
    print('a1=%.4f  min first-negative order=%s  (orders seen %s)  %.0fs'%(a1,mn,sorted(best)[:6],time.time()-t0),flush=True)
json.dump({str(k):v for k,v in out.items()},open('data/scan2_%s.json'%('_'.join(sys.argv[2:])),'w'),default=float)
