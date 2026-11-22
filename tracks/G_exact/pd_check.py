"""Fast rigorous positive-definiteness checks for exact rational symmetric matrices.
rump(): float64 Cholesky of fl(Q) - c I with Rump's bound (BIT 2006): if Cholesky succeeds then Q is PD.
arb_ldl(): outward-rounded ball LDL^T at `bits` precision; all pivot balls > 0 proves PD.
"""
import time,math
import numpy as np,scipy.linalg as la
from fractions import Fraction
from flint import arb,ctx

def rump(Q):
    n=len(Q);t=time.time()
    for i in range(n):
        for j in range(i):
            if Q[i][j]!=Q[j][i]:raise RuntimeError('not symmetric')
    A=np.array([[float(v) for v in row] for row in Q])
    u=2.0**-53
    # |fl(Q)-Q| <= u|Q| entrywise => ||E||_2 <= u*|| |A| ||_2 <= u*||A||_F-abs ; use |A|_2 <= sqrt(max row sum * max col sum) (symmetric: max row sum)
    absnorm=np.abs(A).sum(axis=1).max()
    alpha=(n+2)*u/(1-2*(n+2)*u)
    tr=np.trace(A)
    c=alpha*tr+u*absnorm*(1+2*u)       # Rump shift + conversion error of Q->float
    c*=1.0000001
    B=A-c*np.eye(n)
    try:
        la.cholesky(B,lower=True,check_finite=True);ok=True
    except la.LinAlgError:ok=False
    return {'proved_pd':bool(ok),'shift_c':c,'seconds':time.time()-t,'method':'rump_float_cholesky'}

def arb_ldl(Q,bits=192):
    old=ctx.prec;ctx.prec=bits;t=time.time()
    try:
        n=len(Q)
        a=[[arb(v.numerator)/v.denominator for v in row] for row in Q];minpiv=None
        for k in range(n):
            p=a[k][k]
            if not p>0:return {'proved_pd':False,'fail_at':k,'seconds':time.time()-t,'method':'arb_ldl'}
            lo=p.lower() if hasattr(p,'lower') else p
            minpiv=p if minpiv is None or p<minpiv else minpiv
            for i in range(k+1,n):
                r=a[i][k]/p
                for j in range(i,n):a[i][j]=a[i][j]-r*a[j][k];a[j][i]=a[i][j] if j!=i else a[i][j]
        return {'proved_pd':True,'min_pivot':str(minpiv),'seconds':time.time()-t,'method':'arb_ldl','bits':bits}
    finally:ctx.prec=old
