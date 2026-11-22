"""GPU double-double dual-barrier Newton solver for SOS Gram feasibility (analytic-center path).
 Find X>=0 (blockdiag) with A vec(X)=b.  X=Z^-1, Z = t I - mat(A^T y); y minimizes phi(y) = -b^T y - logdet Z(y)
 (Newton: M dy = -g, g = A vec(X) - b, M = A (X (x) X) A^T).  FP64 phase then DD phase (DD residuals, DD inverse via
 Newton-Schulz, FP64 Cholesky of M + DD-residual iterative refinement).
Usage: solve.py m [--t T] [--out seed.json]"""
import os,sys,time,json,math,argparse
import numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dd,ddla
ap=argparse.ArgumentParser();ap.add_argument("m",type=int);ap.add_argument("--t",type=float,default=1e-4)
ap.add_argument("--out",default=None);ap.add_argument("--maxit",type=int,default=300);ap.add_argument("--tol",type=float,default=1e-17)
ap.add_argument("--cg",type=int,default=20);ap.add_argument("--p1",type=int,default=400);ap.add_argument("--lam1",type=float,default=0.3);ap.add_argument("--delta",type=str,default="0");ap.add_argument("--f64only",action="store_true");ap.add_argument("--resume",default=None);ap.add_argument("--tag",default="")
a=ap.parse_args();m=a.m;t=a.t
HERE=os.path.dirname(os.path.abspath(__file__));D=np.load(f"{HERE}/data/dd_m{m}.npz");dev="cuda"
dims=[int(v) for v in D["dims"]];norm=int(D["norm"]);scales=[D["scales0"].tolist(),D["scales1"].tolist()]
offs=[0,dims[0]**2];N=sum(n*n for n in dims);R=len(D["bh"])
def T(x):return torch.from_numpy(np.ascontiguousarray(x)).to(dev)
rr=T(D["R"]).long();cc=T(D["C"]).long()
AH=torch.zeros(R,N,dtype=torch.float64,device=dev);AL=torch.zeros_like(AH);AH[rr,cc]=T(D["H"]);AL[rr,cc]=T(D["L"]);del rr,cc
b=(T(D["bh"]),T(D["bl"]));I_b=[torch.eye(n,dtype=torch.float64,device=dev) for n in dims]
tt=time.time();SA=dd.split_last(AH,AL,N);SAt=dd.split_last(AH.t().contiguous(),AL.t().contiguous(),R);torch.cuda.synchronize()
print(f"m={m} R={R} N={N} dims={dims} t={t} slices A:{len(SA[1])} At:{len(SAt[1])} setup {time.time()-tt:.1f}s",flush=True)
def log(*s):print(f"[{time.time()-T0:7.1f}s]",*s,flush=True)
T0=time.time()
# ---- DD operators
def Aty_dd(y):  # y DD [R] -> DD [N]
    sy=dd.split_last(y[0].view(1,R),y[1].view(1,R),R);r=dd.mm_split(SAt,sy);return (r[0].view(N),r[1].view(N))
def Ax_dd(x):   # DD [N] -> DD [R]
    sx=dd.split_last(x[0].view(1,N),x[1].view(1,N),N);r=dd.mm_split(SA,sx);return (r[0].view(R),r[1].view(R))
from fractions import Fraction
_vi=torch.cat([I_b[k].reshape(-1) for k in range(2)]);_AI=Ax_dd(dd.from_f(_vi));_c=Fraction(1,100000)-Fraction(a.delta);_ch=float(_c);_cl=float(_c-Fraction(_ch))
_cc=(torch.full((R,),_ch,dtype=torch.float64,device=dev),torch.full((R,),_cl,dtype=torch.float64,device=dev))
b=dd.add(b,dd.mul(_AI,_cc))   # b_delta = b_orig - delta*A vec(I)  (file b already had -1e-5*A vec(I))
DELTA=Fraction(a.delta)
def blocks(v):return [(v[0][o:o+n*n].view(n,n),v[1][o:o+n*n].view(n,n)) for o,n in zip(offs,dims)]
def unblocks(bl):return (torch.cat([x[0].reshape(-1) for x in bl]),torch.cat([x[1].reshape(-1) for x in bl]))
def Zdd(y,tt_=None):
    tt_=t if tt_ is None else tt_
    bl=blocks(Aty_dd(y));return [dd.sub(dd.from_f(tt_*I_b[k]),bl[k]) for k in range(2)]
def dd_inv(Z,verbose=False):
    X0=torch.linalg.inv(Z[0]);X0=(X0+X0.t())/2;X=(X0,torch.zeros_like(X0));n=Z[0].shape[0];Id=dd.from_f(torch.eye(n,dtype=torch.float64,device=dev));prev=1e300
    for it in range(10):
        Rr=dd.sub(Id,dd.matmul(Z,X));rn=float(Rr[0].abs().max())
        if rn>1.5 and it==0:raise RuntimeError(f"Newton-Schulz start failed rn={rn}")
        X=dd.add(X,dd.matmul(X,Rr))
        if rn<1e-16 or (rn>0.5*prev and rn<1e-12):break
        prev=rn
    X=dd.mulf(dd.add(X,(X[0].t().contiguous(),X[1].t().contiguous())),0.5)
    return X,rn
def schur64(Xh):
    M=torch.zeros(R,R,dtype=torch.float64,device=dev)
    for o,n,X in zip(offs,dims,Xh):
        Ab=AH[:,o:o+n*n].reshape(R,n,n);W=X@Ab@X;M+=Ab.reshape(R,-1)@W.reshape(R,-1).t()
    return (M+M.t())/2
SAb=None
def schur_dd(Xb):
    global SAb
    if SAb is None:
        SAb=[dd.split_last(AH[:,o:o+n*n].reshape(R,n,n).contiguous(),AL[:,o:o+n*n].reshape(R,n,n).contiguous(),n) for o,n in zip(offs,dims)]
    Ws=[]
    for k,n in enumerate(dims):
        X=Xb[k];SX=dd.split_last(X[0],X[1],n)
        Tm=dd.mm_split(SAb[k],SX)                       # Ab X   [R,n,n]
        ST=dd.split_last(Tm[0].transpose(-1,-2).contiguous(),Tm[1].transpose(-1,-2).contiguous(),n)
        Wk=dd.mm_split(SX,ST)                           # X (Ab X)
        Ws.append((Wk[0].reshape(R,-1),Wk[1].reshape(R,-1)))
    Wh=torch.cat([w[0] for w in Ws],1);Wl=torch.cat([w[1] for w in Ws],1)
    SW=dd.split_last(Wh,Wl,N);Md=dd.mm_split(SA,SW)
    Mt=(Md[0].t().contiguous(),Md[1].t().contiguous())
    return dd.mulf(dd.add(Md,Mt),0.5)
def Mv_dd(Xdd,dy):
    bl=blocks(Aty_dd(dy));out=[]
    for k in range(2):
        out.append(dd.matmul(dd.matmul(Xdd[k],bl[k]),Xdd[k]))
    return Ax_dd(unblocks(out))
def chol_factor(M):
    d=M.diagonal().clamp(min=1e-300).rsqrt();Ms=M*d[:,None]*d[None,:]
    L,info=torch.linalg.cholesky_ex(Ms)
    if int(info)!=0:
        for j in (1e-14,1e-12,1e-10):
            L,info=torch.linalg.cholesky_ex(Ms+j*torch.eye(R,dtype=torch.float64,device=dev))
            if int(info)==0:break
        else:raise RuntimeError("Schur Cholesky failed")
    return L,d
def chol_solve(F,r):
    L,d=F;return (torch.cholesky_solve((r*d).view(R,1),L).view(R))*d
def logdet_f(Zh):
    s=0.0
    for Z in Zh:
        L,info=torch.linalg.cholesky_ex(Z)
        if int(info)!=0:return None
        s+=2*float(torch.log(L.diagonal()).sum())
    return s
def phi_f(yh):
    Zh=[(t*I_b[k]-blocks((torch.mv(AH.t(),yh),torch.zeros(N,device=dev,dtype=torch.float64)))[k][0]) for k in range(2)]
    ld=logdet_f(Zh)
    return None if ld is None else -float(torch.dot(b[0],yh))-ld
# ---- phase 1: FP64 damped Newton
y=(torch.zeros(R,dtype=torch.float64,device=dev),torch.zeros(R,dtype=torch.float64,device=dev))
if a.resume:
    ck=np.load(a.resume);y=(T(ck["yh"]),T(ck["yl"]))
yh=y[0].clone();it=0
def f64_state(yh):
    Zh=[t*I_b[k]-blocks((torch.mv(AH.t(),yh),torch.zeros(N,device=dev,dtype=torch.float64)))[k][0] for k in range(2)]
    Xh=[torch.linalg.inv(Z) for Z in Zh];Xh=[(X+X.t())/2 for X in Xh]
    g=torch.mv(AH,torch.cat([X.reshape(-1) for X in Xh]))-b[0]
    return Zh,Xh,g
phase1_done=a.resume is not None
while not phase1_done and it<a.p1:
    it+=1;Zh,Xh,g=f64_state(yh);M=schur64(Xh);F=chol_factor(M);dy=-chol_solve(F,g)
    lam2=-float(torch.dot(g,dy));f0=phi_f(yh);alpha=1.0
    while True:
        f1=phi_f(yh+alpha*dy)
        if f1 is not None and f1<=f0-1e-4*alpha*lam2:break
        alpha*=0.5
        if alpha<1e-10:break
    yh=yh+alpha*dy
    log(f"f64 it{it} |g|inf={float(g.abs().max()):.3e} lam2={lam2:.3e} alpha={alpha:.3g} phi={f0:.8e}")
    if alpha<1e-9 or lam2<a.lam1:phase1_done=True
y=(yh,torch.zeros_like(yh))
if a.f64only:
    sys.exit(0)
# ---- phase 2: DD
def save_ck(tag):
    np.savez(f"{HERE}/data/ck_m{m}{a.tag}.npz",yh=y[0].cpu().numpy(),yl=y[1].cpu().numpy(),t=t)
hist=[]
for it2 in range(1,a.maxit+1):
    t1=time.time();Z=Zdd(y);Xb=[];rns=[]
    for k in range(2):
        X,rn=dd_inv(Z[k]);Xb.append(X);rns.append(rn)
    x=unblocks(Xb);g=dd.sub(Ax_dd(x),b);gn=float(g[0].abs().max())
    eigs=[torch.linalg.eigvalsh(X[0]) for X in Xb];emin=[float(e[0]) for e in eigs];emax=[float(e[-1]) for e in eigs]
    tM=time.time();Md=schur_dd(Xb);torch.cuda.synchronize();tM=time.time()-tM;tC=time.time();Ld=ddla.cholesky(Md);torch.cuda.synchronize();tC=time.time()-tC
    rhs=dd.neg(g);dy=ddla.chol_solve(Ld,rhs);rcount=1;rn_=1.0
    for rf in range(0 if -float(torch.dot(g[0],dy[0]))>1.5 else 3):
        Mdy=dd.matmul(Md,(dy[0].view(R,1),dy[1].view(R,1)));r=dd.sub(rhs,(Mdy[0].view(R),Mdy[1].view(R)))
        rn_=float(r[0].abs().max())/max(float(rhs[0].abs().max()),1e-300)
        if rn_<1e-24:break
        dy=dd.add(dy,ddla.chol_solve(Ld,r));rcount+=1
    log(f"   timings M={tM:.1f}s chol={tC:.1f}s")
    lam2=-float(torch.dot(g[0],dy[0]));lam=math.sqrt(max(lam2,0))
    hist.append(gn);log(f"dd it{it2} |g|inf={gn:.3e} lam2={lam2:.3e} NSres={rns[0]:.1e},{rns[1]:.1e} Xmin={emin} Xmax=({emax[0]:.3e},{emax[1]:.3e}) refit={rcount} rel_ref={rn_:.1e} t={time.time()-t1:.1f}s")
    if gn<a.tol:break
    alpha=1.0;f0=phi_f(y[0])
    while True:
        ynew=dd.add(y,dd.mulf(dy,alpha));f1=phi_f(ynew[0])
        if f1 is not None and (lam<1e-2 or f1<=f0-1e-4*alpha*lam2):break
        alpha*=0.5
        if alpha<1e-12:raise RuntimeError("step failed")
    log(f"   alpha={alpha:.3g} phi={f0:.10e}")
    y=ynew
    if it2%5==0:save_ck("")
save_ck("")
# ---- final: X, Q = X + 1e-5 I, gram strings
Z=Zdd(y);Xb=[dd_inv(Z[k])[0] for k in range(2)];x=unblocks(Xb);g=dd.sub(Ax_dd(x),b)
log(f"pre-projection |g|inf={float(g[0].abs().max()):.3e}")
# DD least-squares projection onto A x = b:  x <- x - A^T (A A^T)^-1 (A x - b)
Gd=dd.mm_split(SA,SA);Gd=dd.mulf(dd.add(Gd,(Gd[0].t().contiguous(),Gd[1].t().contiguous())),0.5);Lg=ddla.cholesky(Gd)
for pit in range(5):
    w=ddla.chol_solve(Lg,g)
    for _ in range(2):
        Gw=dd.matmul(Gd,(w[0].view(R,1),w[1].view(R,1)));rr_=dd.sub(g,(Gw[0].view(R),Gw[1].view(R)));w=dd.add(w,ddla.chol_solve(Lg,rr_))
    x=dd.sub(x,Aty_dd(w));g=dd.sub(Ax_dd(x),b);gi=float(g[0].abs().max())
    log(f"projection {pit}: |w|inf={float(w[0].abs().max()):.3e} residual |g|inf={gi:.3e}")
    if gi<1e-29:break
Xb=[(x[0][o:o+n*n].view(n,n).clone(),x[1][o:o+n*n].view(n,n).clone()) for o,n in zip(offs,dims)]
log(f"FINAL |g|inf={float(g[0].abs().max()):.3e} (hi part; lo~{float(g[1].abs().max()):.1e}) |b|inf={float(b[0].abs().max()):.3e} trace={[float(torch.trace(X[0])) for X in Xb]}")
eigs=[]
for X in Xb:
    Q=X[0]+float(DELTA)*torch.eye(X[0].shape[0],dtype=torch.float64,device=dev);eigs.append(float(torch.linalg.eigvalsh(Q)[0]))
log(f"min eig Q_scaled={eigs}  min eig X={[float(torch.linalg.eigvalsh(X[0])[0]) for X in Xb]}")
if a.out:
    import mpmath
    mpmath.mp.prec=200;grams=[]
    for k,(X,n) in enumerate(zip(Xb,dims)):
        Xh=X[0].cpu().numpy();Xl=X[1].cpu().numpy();s=scales[k];G=[]
        for i in range(n):
            row=[]
            for j in range(n):
                v=mpmath.mpf(float(Xh[i,j]))+mpmath.mpf(float(Xl[i,j]))+mpmath.mpf(DELTA.numerator)/mpmath.mpf(DELTA.denominator)*(1 if i==j else 0)
                row.append(mpmath.nstr(v*norm/(s[i]*s[j]),40))
            G.append(row)
        grams.append(G)
    r={"order":m,"status":"dd_dual_barrier_newton","proved":False,"normalization":norm,"basis_scales":scales,"min_scaled_eigenvalues":eigs,
       "residual":float(g[0].abs().max()),"solver":"GPU double-double dual-barrier Newton (tracks/E_ddipm)","t":t,"gram_matrices":grams}
    json.dump(r,open(a.out,"w"));log("wrote",a.out)
