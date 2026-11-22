import sys; sys.path.insert(0,'/tmp')
from kinks import *
import mpmath as mp
mp.mp.dps=110
def q_mp(x, xs, ss):
    xs_=[mp.mpf(a) for a in xs]; ss_=[mp.mpf(a) for a in ss]
    n=len(xs_); c=[mp.mpf(0)]
    for i in range(n): c.append(c[i]+(ss_[i]-ss_[i+1])*xs_[i])
    bl=[-mp.inf]+xs_; bu=xs_+[mp.inf]; tot=0
    for i in range(n+1):
        tot += mp.exp(-c[i]-ss_[i]*x+ss_[i]**2/2)*(mp.ncdf(bu[i]+ss_[i]-x)-mp.ncdf(bl[i]+ss_[i]-x))
    return tot
cfgs=[([-1.5,1.5],[-1,0,1]),([-1,0.5,2],[-1.5,-0.4,0.3,1.2])]
for xs,ss in cfgs:
    h=0.25
    x,h=grid_for(xs,ss,margin=10,h=h)
    Q,L=jets(x,xs,ss); u=logderivs(Q)
    q=Q[0]*np.exp(L-L.max()); 
    # mp: same grid, weights q_mp
    mpu=[]; mpq=[]
    for xv in x:
        xm=mp.mpf(float(xv))
        t=mp.taylor(lambda y: mp.log(q_mp(y,xs,ss)), xm, 16, h=mp.mpf('1e-25'))
        mpu.append([None]+[t[k]*mp.factorial(k) for k in range(1,17)]); mpq.append(q_mp(xm,xs,ss))
    for m in [6,10,13,14,15,16]:
        Sf=sum(INTEGRANDS[m](tuple([None]+[u[k][j] for k in range(1,17)])) * Q[0][j]*np.exp(L[j]) for j in range(len(x)))
        Sm=sum(INTEGRANDS[m](mpu[j])*mpq[j] for j in range(len(x)))
        print(xs,ss,'m',m,'fp64 %.12e'%Sf,'mp',mp.nstr(Sm,14),'relerr',mp.nstr(abs(Sf-Sm)/abs(Sm),3),flush=True)
