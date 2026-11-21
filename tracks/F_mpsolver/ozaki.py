"""Ozaki-scheme DD-accurate matmul.  Two variants:
  fp64 : slices with beta=(51-log2 k)/2 bits, exact FP64 GEMMs  (tracks/E_ddipm/dd.py: split_last/mm_split)
  int8 : slices of BITS=6 bits as int8, exact INT32 accumulation on tensor cores (torch._int_mm), diag-grouped sums.
A: DD [m,k], B: DD [k,n]  -> DD [m,n] (relative accuracy ~2^-106 w.r.t. row/col scale of the operands)."""
import os,sys,math,torch
sys.path.insert(0,os.path.expanduser("~/gcm-logconcave-counterexamples/tracks/E_ddipm"))
import dd
BITS=6
def _pad8(n):return (n+7)//8*8
def i8_split(Mh,Ml,L=None,total_bits=112):
    """Slice DD tensor [...,r,k] along last axis. Returns (sigma[...,r,1], slices int8 list, L)."""
    L=L or math.ceil(total_bits/BITS)
    mx=Mh.abs().amax(dim=-1,keepdim=True)
    e=torch.ceil(torch.log2(mx.clamp(min=1e-300)));e=torch.where(mx>0,e,torch.zeros_like(e))
    sigma=torch.exp2(e);rh=Mh/sigma;rl=Ml/sigma;sl=[]
    for s in range(L):
        d=torch.round(rh*(1<<BITS));sl.append(d.to(torch.int8))
        rh,rl=dd.two_sum(rh-d/(1<<BITS),rl);rh=rh*(1<<BITS);rl=rl*(1<<BITS)
    return sigma,sl,L
def i8_prep_left(sl):
    return [s.contiguous() for s in sl]
def i8_mm_split(SA,SBt):
    """SA = i8_split(A [m,k]); SBt = i8_split(B^T [n,k]).  Returns DD [m,n]."""
    sA,slA,L=SA;sB,slB,_=SBt
    m,k=slA[0].shape;n=slB[0].shape[0]
    # pad k to multiple of 8, n to multiple of 8 (via cat), m >16
    kp=_pad8(k);npd=(n+31)//32*32;mp=_pad8(m)
    def padk(x):return torch.nn.functional.pad(x,(0,kp-k)) if kp!=k else x
    Bcat=torch.cat([torch.nn.functional.pad(padk(s),(0,0,0,npd-n)) if npd!=n else padk(s) for s in slB],0)  # [L*npd,kp]
    S=[None]*(2*L+2)  # diag sums as fp64 (exact: k*64*64*L << 2^53)
    for p in range(1,L+1):
        At=padk(slA[p-1])
        if mp!=m:At=torch.nn.functional.pad(At,(0,0,0,mp-m))
        At=At.t().contiguous()  # [kp, mp]
        nq=L+1-p
        P=torch._int_mm(Bcat[:nq*npd],At)  # int32 [nq*npd, mp] = (B_q^T-slices) x A_p^T
        for q in range(1,nq+1):
            blk=P[(q-1)*npd:(q-1)*npd+n,:m].t().double()
            d=p+q;S[d]=blk if S[d] is None else S[d]+blk
    acc=None
    for d in range(L+1,1,-1):
        t=S[d]*(2.0**(-BITS*d))
        z=torch.zeros_like(t)
        acc=(t,z) if acc is None else dd.add(acc,(t,z))
    sc=sA*sB.t()
    return (acc[0]*sc,acc[1]*sc)
def i8_matmul(A,B):
    SA=i8_split(A[0],A[1]);SB=i8_split(B[0].t().contiguous(),B[1].t().contiguous());return i8_mm_split(SA,SB)
