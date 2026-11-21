import sys,os,time,pickle; sys.path.insert(0,os.path.expanduser('~/gcm-logconcave-counterexamples/src'))
import numpy as np
from entropy_sparse import polynomials
from kinks import setup, jets, logderivs, grid_for
_cache=os.path.expanduser('~/gcm-logconcave-counterexamples/tracks/C_kinks/data/polys20.pkl')
def load_polys(M=20):
    if os.path.exists(_cache): return pickle.load(open(_cache,'rb'))
    P=polynomials(M)
    out={m:[(e,c) for e,c in P[m].items()] for m in P}   # exact Fractions
    pickle.dump(out,open(_cache,'wb')); return out
POLY=None
def F_eval(m,u):
    global POLY
    if POLY is None: POLY=load_polys()
    tot=0.0
    for e,c in POLY[m]:
        t=float(c)
        for j,pw in enumerate(e):
            if pw: t=t*u[j+1]**pw
        tot=tot+t
    return tot
def S_R_ext(xs,ss,ms=range(14,21),h=0.05,margin=13.0,K=20):
    p=setup(xs,ss); x,h=grid_for(xs,ss,margin,h)
    Q,L=jets(x,xs,ss,K,p); u=logderivs(Q,K)
    q=Q[0]*np.exp(L-L.max()); q=q/(q.sum()*h)
    out={}
    for m in ms:
        F=F_eval(m,u); S=(-1)**m*np.sum(F*q)*h; A=np.sum(np.abs(F)*q)*h
        out[m]=(float(S),float(A),float(S/A))
    return out
if __name__=='__main__':
    t=time.time(); P=load_polys(); print('polys',{m:len(P[m]) for m in (15,17,20)},time.time()-t,flush=True)
    for a in [0.3,1,3]:
      for d in [3,50]:
        r=S_R_ext([-d/2,d/2],[-a,0,a],ms=range(14,21))
        print('a',a,'d',d,' '.join('%d:%+.2e'%(m,r[m][2]) for m in r),flush=True)
