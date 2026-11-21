import json,sys
from ts import *
from an2 import loadg
mp.mp.dps=40
G=mp.gamma(mp.mpf(-3)/2)
def cands(name,xtol=1e-6):
    d=json.load(open('data/sing_%s.json'%name)); out=[]
    cfg=CONFIGS[name]; a=cfg['a']; kap=cfg['kappa']
    for o in d:
        tau=mp.mpc(*o['tau']); x=mp.mpc(*o['x']); sig=-tau
        al,be=local_data(a,tau,x)
        q=2*al*sig/be
        re=float(x.real); mult=1 if (abs(re)<1e-6 or abs(re-math.pi)<1e-6) else 2
        T=sig/(1+kap*sig)
        out.append(dict(sig=sig,x=x,T=T,q=q,be=be,al=al,mult=mult,realx=abs(float(x.imag))<xtol))
    return out
def cpred(c,eps=1):
    K=(c['be']/3)*eps*mp.sqrt(c['q'])**3
    return c['mult']*K/G           # g_n ~ c * sig^-n n^{-5/2}
def gpred(cs,n,eps=None):
    return sum(cpred(c,1 if eps is None else eps[i])*c['sig']**(-n)*mp.mpf(n)**(-2.5) for i,c in enumerate(cs))
if __name__=="__main__":
    name=sys.argv[1]; g=loadg(name); cs=cands(name)
    R=[c for c in cs if c['realx']]; R.sort(key=lambda c:abs(c['sig']))
    print(name,'real-x candidates (sigma=-tau*):')
    for c in R[:12]: print('  sigma=%s x=%s |sigma|=%.4f'%(mp.nstr(c['sig'],7),mp.nstr(c['x'].real,6),abs(c['sig'])))
    for K in (1,2,3,4,6,8):
        sub=R[:K]
        print(' first',K,'real-x cands: rel err of g_n at n=40,80,120:',[mp.nstr(abs(gpred(sub,n)-g[n])/abs(g[n]),3) for n in (40,80,120)])
