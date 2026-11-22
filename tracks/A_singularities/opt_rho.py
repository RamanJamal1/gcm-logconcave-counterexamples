"""Hill-climb: maximise rho(a)=kappa*/kappa_min at order m (kappa* = largest kappa with S_m(kappa)<0; kappa_min=max (log r)'' = log-concavity bound).
rho>=1 would give S_m<0 at a log-concave witness (periodic surrogate, float64 screen only). Start points: witness / R15opt."""
import numpy as np, math, sys, json
from scan import logconc, X
def gseq(a,M):
    K=len(a); n=np.arange(1,K+1); c=np.cos(np.outer(n,X)); A=np.array(a)
    Rk=[((A*(n**2/2.0)**k/math.factorial(k))[:,None]*c).sum(0) for k in range(M+1)]
    R0=1+Rk[0]
    if R0.min()<=0: return None
    Rk[0]=R0; L=[np.log(R0)]
    for m in range(1,M+1): L.append((m*Rk[m]-sum(k*L[k]*Rk[m-k] for k in range(1,m)))/(m*R0))
    return [0.0]+[sum(Rk[k]*L[m-k] for k in range(m+1)).mean() for m in range(1,M+1)]
def Sm(g,kap,m): return math.factorial(m)*sum(math.comb(m-1,j-1)*kap**(m-j)*g[j] for j in range(1,m+1))+math.factorial(m-1)*kap**m/2
def rho(a,m):
    km=logconc(a)
    if km<=0: return 0.0,None
    g=gseq(a,m)
    if g is None: return 0.0,None
    ks=km*np.linspace(0.02,1.6,160); v=[Sm(g,k,m) for k in ks]
    neg=[i for i in range(len(ks)) if v[i]<0]
    if not neg: return 0.0,g
    i=max(neg)
    lo,hi=ks[i],ks[min(i+1,len(ks)-1)]
    for _ in range(30):
        mid=(lo+hi)/2
        if Sm(g,mid,m)<0: lo=mid
        else: hi=mid
    return lo/km,g
if __name__=="__main__":
    m=int(sys.argv[1]); seed=int(sys.argv[2]); iters=int(sys.argv[3]); rng=np.random.default_rng(seed)
    starts={'witness3':[1/15,1/40000,-1/18e6],'R15opt5':[0.05581601564460076,1.8605329413968507e-05,-3.985583229780961e-08,2.6536911012714195e-10,-2.852471396848522e-12]}
    out={}
    for nm,a0 in starts.items():
        a=list(a0); r,_=rho(a,m); best=(r,list(a)); step=0.1
        for it in range(iters):
            b=[]
            for k,v in enumerate(best[1]):
                if rng.random()<0.5: b.append(v*math.exp(rng.normal()*step))
                else: b.append(v)
            if rng.random()<0.15: b.append(rng.choice([-1,1])*abs(b[-1])*10**rng.uniform(-3,-1.5)) if len(b)<7 else None
            for k in range(1,len(b)):
                if rng.random()<0.03: b[k]=-b[k]
            r,_=rho(b,m)
            if r>best[0]: best=(r,b)
            if it%2000==1999: step=max(0.01,step*0.8)
        out[nm]=best; print(nm,'m=%d'%m,'best rho=%.4f'%best[0],'a=',best[1],flush=True)
    json.dump(out,open('data/optrho_m%d_s%d.json'%(m,seed),'w'))
