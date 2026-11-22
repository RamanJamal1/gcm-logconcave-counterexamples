"""Exact Dirichlet/'mode-lattice' spectrum: g(sigma)=<R log R> = sum_d C_d e^{d sigma/2}  (d = sum n_i^2 over the factors),
so P_n = n! g_n = sum_d C_d (d/2)^n  and S_m = (1/2)(m-1)! kappa^m + sum_d C_d Kern_m(d/2),
Kern_m(lam)= m! sum_{n=1}^m C(m-1,n-1) kappa^{m-n} lam^n / n!  (>0 for lam>0).  Terms j>=2 of (1+u)log(1+u)=u+sum (-1)^j u^j/(j(j-1))."""
import numpy as np, sys, json
from ts import *
mp.mp.dps=40
def spectrum(a,Dmax=60):
    K=len(a); E=Dmax+1
    off=E   # e index offset
    shape=(Dmax+1,2*E+1)
    def zeros(): 
        z=np.empty(shape,dtype=object); z[:]=mp.mpf(0); return z
    U=zeros(); U[0,off]=mp.mpf(1)   # U_0=1
    C=[mp.mpf(0)]*(Dmax+1)
    for j in range(1,Dmax+1):
        V=zeros()
        for n in range(1,K+1):
            w=a[n-1]/2; d=n*n
            if d>Dmax: continue
            for s in (1,-1):
                src=U[0:Dmax+1-d,:]
                if s==1: V[d:,n:]+=w*src[:,:2*E+1-n]
                else: V[d:,:2*E+1-n]+=w*src[:,n:]
        U=V
        if j>=2:
            c=mp.mpf(-1)**j/(j*(j-1))
            for d in range(Dmax+1): C[d]+=c*U[d,off]
    return C
def Kern(m,lam,kappa): return mp.factorial(m)*sum(comb(m-1,n-1)*kappa**(m-n)*lam**n/mp.factorial(n) for n in range(1,m+1))
if __name__=="__main__":
    name=sys.argv[1]; Dmax=int(sys.argv[2]); cfg=CONFIGS[name]; a=cfg['a']; kap=cfg['kappa']
    C=spectrum(a,Dmax)
    json.dump([mp.nstr(c,25) for c in C],open('data/spectrum_%s_D%d.json'%(name,Dmax),'w'))
    from an2 import loadg
    g=loadg(name,120,256)
    print(name,'check P_n=sum C_d (d/2)^n vs exact (rel err):',[(n,mp.nstr(abs(sum(C[d]*(mp.mpf(d)/2)**n for d in range(1,Dmax+1))-mp.factorial(n)*g[n])/abs(mp.factorial(n)*g[n]),3)) for n in (2,6,10,14,16)])
    print('nonzero C_d (d, lambda=d/2, C_d):')
    print(' '.join('%d:%s'%(d,mp.nstr(C[d],3)) for d in range(1,Dmax+1) if abs(C[d])>0))
    for m in (14,15,16,17):
        tot=mp.factorial(m-1)*kap**m/2; neg=mp.mpf(0); pos=tot
        contrib=[(d,C[d]*Kern(m,mp.mpf(d)/2,kap)) for d in range(1,Dmax+1) if abs(C[d])>0]
        pos+=sum(v for d,v in contrib if v>0); neg=sum(v for d,v in contrib if v<0)
        print('m=%d: Gauss %.3e, positive spectral mass %.3e, negative spectral mass %.3e, sum %.4e | top negative d:'%(m,float(mp.factorial(m-1)*kap**m/2),float(pos-mp.factorial(m-1)*kap**m/2),float(neg),float(pos+neg)),
          sorted([(d,float(v)) for d,v in contrib if v<0],key=lambda t:t[1])[:3])
