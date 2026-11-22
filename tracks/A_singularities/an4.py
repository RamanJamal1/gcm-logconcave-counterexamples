"""Nested-set prediction test for g_n: predictions with ALL candidates (R=R_x=0 double zeros) with |sigma|<=r, A_j=+A_pred (eps=+1), excluding those whose fit ratio ~0 (non-principal)."""
import json,sys
from ts import *
from an2 import loadg
mp.mp.dps=60
def cs(s,n): return mp.gamma(n-s)/(mp.gamma(-s)*mp.factorial(n))
def load_cands(name):
    cfg=CONFIGS[name]; a=cfg['a']; kap=cfg['kappa']; d=json.load(open('data/sing_%s.json'%name)); seen={}
    for o in d:
        tau=mp.mpc(*o['tau']); x=mp.mpc(*o['x']); sig=-tau
        al,be=local_data(a,tau,x); q=2*al*sig/be
        re=float(canon_x(complex(x)).real); mu=1 if (abs(re)<1e-6 or abs(re-math.pi)<1e-6) else 2
        key=(round(float(sig.real),5),round(float(sig.imag),5))
        A=mu*(be/3)*mp.sqrt(q)**3
        if key in seen: seen[key]['A']+=A; seen[key]['xs'].append(complex(x))
        else: seen[key]=dict(sig=sig,A=A,xs=[complex(x)])
    return sorted(seen.values(),key=lambda c:abs(c['sig']))
def gpred(cs_,n): return sum(c['A']*c['sig']**(-n)*cs(mp.mpf(3)/2,n) for c in cs_)
if __name__=="__main__":
    name=sys.argv[1]; skip=[float(v) for v in sys.argv[2:]]   # |sigma| values of known non-principal candidates to skip
    g=loadg(name,400,128); C=load_cands(name)
    C=[c for c in C if not any(abs(abs(c['sig'])-s)<2e-3 for s in skip)]
    for r in [1.0,1.02,1.05,1.1,1.2,1.4,1.7]:
        sub=[c for c in C if abs(c['sig'])<=r*abs(C[0]['sig'])]
        print(name,'|sigma|<=%.2f*%.4f: %d cands'%(r,abs(C[0]['sig']),len(sub)),'rel err n=100,200,300,400:',[mp.nstr(abs(gpred(sub,n)-g[n])/abs(g[n]),3) for n in (100,200,300,400)])
