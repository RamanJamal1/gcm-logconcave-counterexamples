"""Darboux (leading-order, 3/2-power) prediction of S_m vs exact periodic-surrogate S_m.
S_m ~ Gamma(m-3/2)/Gamma(-3/2) * sum_j A'_j T_j^{-m} + (1/2)(m-1)! kappa^m,
T_j = sigma_j/(1+kappa sigma_j),  A'_j = A_j (1+kappa sigma_j)^{3/2},  A_j = mult*(beta/3)*sqrt(2 alpha sigma_j/beta)^3,
(sigma=-tau; alpha=R_tau, beta=R_xx at the double zero R=R_x=0). Candidates with |Im x*|<imx (near-real double zeros) are used; others are
off the principal sheet (see fit_sing.py)."""
import json,sys
from ts import *
from an2 import loadg
mp.mp.dps=60
G32=mp.gamma(mp.mpf(-3)/2)
def cand(name,imx=0.1,rel=1.6):
    cfg=CONFIGS[name]; a=cfg['a']; kap=cfg['kappa']; d=json.load(open('data/sing_%s.json'%name)); seen={}
    for o in d:
        tau=mp.mpc(*o['tau']); x=mp.mpc(*o['x'])
        if abs(float(x.imag))>=imx: continue
        sig=-tau; al,be=local_data(a,tau,x); q=2*al*sig/be
        re=float(canon_x(complex(x)).real); mu=1 if (abs(re)<1e-6 or abs(re-math.pi)<1e-6) else 2
        key=(round(float(sig.real),5),round(float(sig.imag),5)); A=mu*(be/3)*mp.sqrt(q)**3
        if key in seen: seen[key]['A']+=A
        else: seen[key]=dict(sig=sig,A=A,x=complex(x),mult=mu)
    C=sorted(seen.values(),key=lambda c:abs(c['sig']/(1+kap*c['sig'])))
    for c in C:
        c['T']=c['sig']/(1+kap*c['sig']); c['Ap']=c['A']*(1+kap*c['sig'])**mp.mpf(1.5)
    return [c for c in C if abs(c['T'])<=rel*abs(C[0]['T'])]
def SD(C,kap,m,gauss=True):
    s=G32**-1*mp.gamma(m-mp.mpf(3)/2)*sum(c['Ap']*c['T']**(-m) for c in C)
    s=s.real if hasattr(s,'real') else s
    return s+(mp.factorial(m-1)*kap**m/2 if gauss else 0)
if __name__=="__main__":
    name=sys.argv[1]; imx=float(sys.argv[2]) if len(sys.argv)>2 else 0.1; rel=float(sys.argv[3]) if len(sys.argv)>3 else 1.6
    g=loadg(name,400,128); kap=CONFIGS[name]['kappa']; C=cand(name,imx,rel)
    print(name,'kappa=%.6g'%kap,'principal-like candidates (sorted by |T|):')
    rows=[]
    for c in C[:10]:
        print('  sigma=%s x=%s T=%s |T|=%.4f arg(1/T... )A\'=%s'%(mp.nstr(c['sig'],6),mp.nstr(c['x'],4),mp.nstr(c['T'],6),abs(c['T']),mp.nstr(c['Ap'],5)))
    lead=[C[0]]
    print('  m  S_exact  S_Darboux(all %d)  relerr  S_Darboux(leading only)  relerr  signs(ex,all,lead)'%len(C))
    for m in list(range(6,41))+[50,60,80,100,150,200,300,399]:
        ex=S_from_h(g,kap,m); p=SD(C,kap,m); p1=SD(lead,kap,m)
        rows.append(dict(m=m,exact=float(ex),darboux=float(p),darboux_lead=float(p1)))
        print('  %3d %11.4e %11.4e %8.3f(m*err=%.1f) %11.4e %8.3f   %s%s%s'%(m,ex,p,abs(p-ex)/abs(ex),m*abs(p-ex)/abs(ex),p1,abs(p1-ex)/abs(ex),'+' if ex>0 else '-','+' if p>0 else '-','+' if p1>0 else '-'))
    json.dump(rows,open('data/darboux_%s.json'%name,'w'),indent=0)
