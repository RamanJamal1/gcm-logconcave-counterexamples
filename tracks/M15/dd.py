"""Double-double (hi,lo float64) arithmetic in torch on CUDA, plus Ozaki-style exact-slice DD matmul.
Elementwise torch ops are separate kernels, so each op rounds individually (no FMA contraction); Dekker split is used."""
import math,torch
from fractions import Fraction
SPLIT=134217729.0

def two_sum(a,b):
    s=a+b;bb=s-a;return s,(a-(s-bb))+(b-bb)
def quick_two_sum(a,b):
    s=a+b;return s,b-(s-a)
def _split(a):
    t=SPLIT*a;hi=t-(t-a);return hi,a-hi
def two_prod(a,b):
    p=a*b;ah,al=_split(a);bh,bl=_split(b)
    return p,((ah*bh-p)+ah*bl+al*bh)+al*bl

def add(x,y):
    s1,s2=two_sum(x[0],y[0]);t1,t2=two_sum(x[1],y[1]);s2=s2+t1
    s1,s2=quick_two_sum(s1,s2);s2=s2+t2;return quick_two_sum(s1,s2)
def neg(x):return (-x[0],-x[1])
def sub(x,y):return add(x,neg(y))
def mul(x,y):
    p,e=two_prod(x[0],y[0]);e=e+(x[0]*y[1]+x[1]*y[0]);return quick_two_sum(p,e)
def mulf(x,f):
    p,e=two_prod(x[0],f);e=e+x[1]*f;return quick_two_sum(p,e)
def div(x,y):
    q1=x[0]/y[0];r=sub(x,mulf(y,q1));q2=r[0]/y[0];r=sub(r,mulf(y,q2));q3=r[0]/y[0]
    q1,q2=quick_two_sum(q1,q2);return add((q1,q2),(q3,torch.zeros_like(q3)))
def sqrt(x):
    r=torch.rsqrt(x[0]);ax=x[0]*r;p,e=two_prod(ax,ax);d=sub(x,(p,e))
    return two_sum(ax,d[0]*r*0.5)
def zeros(*shape,device="cuda"):
    z=torch.zeros(*shape,dtype=torch.float64,device=device);return (z,z.clone())
def from_f(a):return (a,torch.zeros_like(a))
def to_fraction(x):
    h=x[0].double().cpu().numpy();l=x[1].double().cpu().numpy();return h,l

# ---------------- Ozaki slicing matmul ----------------
def _params(k,bits=112):
    beta=(51-math.ceil(math.log2(max(k,2))))//2;L=math.ceil(bits/beta);return beta,L

def split_last(Mh,Ml,k,bits=112):
    """Slice a DD tensor along its last axis (the contraction axis). Returns (sigma, [slices])."""
    beta,L=_params(k,bits)
    mx=Mh.abs().amax(dim=-1,keepdim=True)
    e=torch.ceil(torch.log2(mx.clamp(min=1e-300)));e=torch.where(mx>0,e,torch.zeros_like(e))
    sigma=torch.exp2(e);rh=Mh/sigma;rl=Ml/sigma
    sl=[]
    for s in range(1,L+1):
        u=2.0**(-beta*s);q=torch.round(rh/u)*u;sl.append(q)
        rh,rl=two_sum(rh-q,rl)
    return sigma,sl,beta

def mm_split(SA,SB):
    """SA from split_last(A) with A [...,m,k]; SB from split_last(B^T) with B^T [...,n,k]. Returns DD [...,m,n]."""
    sA,slA,beta=SA;sB,slB,_=SB;L=len(slA)
    acc=None
    # accumulate small magnitudes first
    for d in range(L+1,1,-1):
        g=None
        for p in range(1,L+1):
            q=d-p
            if q<1 or q>L:continue
            P=torch.matmul(slA[p-1],slB[q-1].transpose(-1,-2))
            g=(P,torch.zeros_like(P)) if g is None else add(g,(P,torch.zeros_like(P)))
        if g is None:continue
        acc=g if acc is None else add(acc,g)
    scale=sA*sB.transpose(-1,-2)
    return (acc[0]*scale,acc[1]*scale)

def matmul(A,B,bits=112):
    """DD matmul of DD tensors A [...,m,k], B [...,k,n]."""
    k=A[0].shape[-1]
    SA=split_last(A[0],A[1],k,bits);SB=split_last(B[0].transpose(-1,-2).contiguous(),B[1].transpose(-1,-2).contiguous(),k,bits)
    return mm_split(SA,SB)

# --------------- host conversions ---------------
def dd_to_fraction(h,l):return Fraction(float(h))+Fraction(float(l))
def dd_to_str(h,l,digits=40):
    import mpmath
    mpmath.mp.prec=200
    return mpmath.nstr(mpmath.mpf(float(h))+mpmath.mpf(float(l)),digits)
