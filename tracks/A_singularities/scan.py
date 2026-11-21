"""Fast float64 periodic-surrogate scan: first negative order of S_m over random multi-mode families, kappa = (1+margin)*max (log r)''.
NUMERICAL SCREEN ONLY (float64, surrogate without Gaussian-filter terms); candidates must be re-verified with mpmath/certifier."""
import numpy as np, math, sys, json, time
G=256; X=2*np.pi*np.arange(G)/G
def S_all(a,kappa,M=24):
    K=len(a); n=np.arange(1,K+1)
    c=np.cos(np.outer(n,X))            # K x G
    R=[]
    for k in range(M+1):
        R.append(((np.array(a)*(n**2/2.0)**k/math.factorial(k))[:,None]*c).sum(0))   # Taylor in sigma=-tau (repo sign)
    R0=1+R[0]
    if R0.min()<=0: return None
    L=[np.log(R0)]
    for m in range(1,M+1):
        L.append((m*R[m]-sum(k*L[k]*R[m-k] for k in range(1,m)))/(m*R0)) if False else None
    # series of R(sigma)=R0+sum R_k s^k with R[0] replaced
    Rk=[R0]+R[1:]
    L=[np.log(R0)]
    for m in range(1,M+1):
        L.append((m*Rk[m]-sum(k*L[k]*Rk[m-k] for k in range(1,m)))/(m*R0))
    g=[0.0]
    for m in range(1,M+1):
        g.append(sum(Rk[k]*L[m-k] for k in range(m+1)).mean())
    S=[0.0]
    for m in range(1,M+1):
        S.append(math.factorial(m)*sum(math.comb(m-1,j-1)*kappa**(m-j)*g[j] for j in range(1,m+1))+math.factorial(m-1)*kappa**m/2)
    return S
def logconc(a):
    n=np.arange(1,len(a)+1); R=1+(np.array(a)[:,None]*np.cos(np.outer(n,X))).sum(0)
    Rp=-(np.array(a)[:,None]*n[:,None]*np.sin(np.outer(n,X))).sum(0); Rpp=-(np.array(a)[:,None]*n[:,None]**2*np.cos(np.outer(n,X))).sum(0)
    return ((Rpp*R-Rp**2)/R**2).max()
if __name__=="__main__":
    seed=int(sys.argv[1]); N=int(sys.argv[2]); rng=np.random.default_rng(seed); best=[]; t0=time.time(); hist={}
    for it in range(N):
        K=rng.integers(2,6); a1=10**rng.uniform(-2.3,-0.7)
        a=[a1]+[rng.choice([-1,1])*10**rng.uniform(-3*k-1,-0.3*k) for k in range(1,K)]
        a=[a[0]]+[a[k]*1.0 for k in range(1,K)]
        lc=logconc(a)
        if lc<=0: continue
        kap=lc*(1+rng.choice([0,0,0.02,0.1,0.3]))
        S=S_all(a,kap)
        if S is None: continue
        neg=[m for m in range(2,25) if S[m]<0]
        if not neg: continue
        f=neg[0]; hist[f]=hist.get(f,0)+1
        best.append((f,a,kap,[S[m] for m in range(10,25)]))
    best.sort(key=lambda r:r[0])
    json.dump({'hist':hist,'best':best[:40]},open('data/scan_seed%d.json'%seed,'w'))
    print(seed,'time',time.time()-t0,'hist',sorted(hist.items()))
