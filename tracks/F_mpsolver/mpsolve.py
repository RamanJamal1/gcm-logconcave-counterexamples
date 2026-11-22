"""Mixed-precision GPU interior-point (dual-barrier Newton) solver for SOS Gram strict feasibility.
 Find X=blkdiag(X0,X1) PD with A vec(X)=b0 (UNSHIFTED, margin 0).  X=Z^-1, Z=t I - mat(A^T y), y minimises
 phi(y)=-b^T y - logdet Z(y)  (analytic-centre-like; mu=t moderate).  Newton system M dy=-g, M=A(X(x)X)A^T, g=A x-b.
 Linear solves: FP64 Cholesky preconditioner (Jacobi-equilibrated) + GMRES-IR with DD residual/matvec (Carson-Higham).
 All feasibility residuals in double-double (Ozaki-sliced FP64 GEMMs).
Usage: mpsolve.py m [--t T]"""
import os,sys,time,json,math,argparse
import numpy as np,torch
HERE=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,HERE)
import ozaki,dd
ap=argparse.ArgumentParser();ap.add_argument("m",type=int);ap.add_argument("--t",type=float,default=1e-4)
ap.add_argument("--maxit",type=int,default=300);ap.add_argument("--tol",type=float,default=1e-18)
ap.add_argument("--p1",type=int,default=400);ap.add_argument("--lam1",type=float,default=0.3)
ap.add_argument("--gm",type=int,default=30);ap.add_argument("--ir_tol",type=float,default=1e-13);ap.add_argument("--ir_max",type=int,default=8)
ap.add_argument("--resume",default=None);ap.add_argument("--tag",default="");ap.add_argument("--gpu",type=int,default=0);ap.add_argument("--f64only",action="store_true")
ap.add_argument("--final_tol",type=float,default=1e-29)
a=ap.parse_args();m=a.m;t=a.t
torch.cuda.set_device(a.gpu);dev="cuda"
D=np.load(f"{HERE}/data/dd_m{m}.npz")
dims=[int(v) for v in D["dims"]];norm=int(D["norm"]);scales=[D["scales0"].tolist(),D["scales1"].tolist()]
offs=[0,dims[0]**2];N=sum(n*n for n in dims);R=len(D["bh"])
def T(x):return torch.from_numpy(np.ascontiguousarray(x)).to(dev)
rr=T(D["R"]).long();cc=T(D["C"]).long()
AH=torch.zeros(R,N,dtype=torch.float64,device=dev);AL=torch.zeros_like(AH);AH[rr,cc]=T(D["H"]);AL[rr,cc]=T(D["L"]);del rr,cc
b=(T(D["bh"]),T(D["bl"]));I_b=[torch.eye(n,dtype=torch.float64,device=dev) for n in dims]
T0=time.time()
def log(*s):print(f"[{time.time()-T0:7.1f}s]",*s,flush=True)
TM={}
def tic():torch.cuda.synchronize();return time.time()
def tadd(k,t0):TM[k]=TM.get(k,0)+tic()-t0
t0=tic();SA=dd.split_last(AH,AL,N);SAt=dd.split_last(AH.t().contiguous(),AL.t().contiguous(),R);tadd("setup",t0)
log(f"m={m} R={R} N={N} dims={dims} t={t} slices A:{len(SA[1])} At:{len(SAt[1])} setup {TM['setup']:.1f}s gpu mem {torch.cuda.memory_allocated()/1e9:.1f}GB")
def gemv(S,v,k):
    """DD gemv: S=split_last(Mat [m,k]) -> (sig [m,1], slices, beta); v DD [k] -> DD [m]. Vector slices stacked so each A-slice is read once."""
    sig,sl,beta=S;L=len(sl)
    sv,slv,_=dd.split_last(v[0].view(1,k),v[1].view(1,k),k)
    V=torch.stack([s.view(-1) for s in slv],1)  # [k,L]
    P=[sl[p]@V[:,:L-p] for p in range(L)]       # P[p][:,q] -> pair (p+1,q+1)
    acc=None
    for d in range(L+1,1,-1):
        g=None
        for p in range(1,d):
            q=d-p
            if q>L or p>L:continue
            x=P[p-1][:,q-1];x=(x,torch.zeros_like(x));g=x if g is None else dd.add(g,x)
        acc=g if acc is None else dd.add(acc,g)
    sc=(sig.view(-1))*sv.view(())
    return (acc[0]*sc,acc[1]*sc)
def Aty_dd(y):return gemv(SAt,y,R)
def Ax_dd(x):return gemv(SA,x,N)
def blocks(v):return [(v[0][o:o+n*n].view(n,n),v[1][o:o+n*n].view(n,n)) for o,n in zip(offs,dims)]
def unblocks(bl):return (torch.cat([x[0].reshape(-1) for x in bl]),torch.cat([x[1].reshape(-1) for x in bl]))
def Zdd(y):
    bl=blocks(Aty_dd(y));return [dd.sub(dd.from_f(t*I_b[k]),bl[k]) for k in range(2)]
def dd_inv(Z):
    X0=torch.linalg.inv(Z[0]);X0=(X0+X0.t())/2;X=(X0,torch.zeros_like(X0));n=Z[0].shape[0];Id=dd.from_f(torch.eye(n,dtype=torch.float64,device=dev));prev=1e300
    for it in range(12):
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
def Mv_dd(Xb,v):
    bl=blocks(Aty_dd(v));out=[dd.matmul(dd.matmul(Xb[k],bl[k]),Xb[k]) for k in range(2)]
    return Ax_dd(unblocks(out))
def chol_factor(M):
    d=M.diagonal().clamp(min=1e-300).rsqrt();Ms=M*d[:,None]*d[None,:];jit=0.0
    L,info=torch.linalg.cholesky_ex(Ms)
    if int(info)!=0:
        for jit in (1e-15,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9):
            L,info=torch.linalg.cholesky_ex(Ms+jit*torch.eye(R,dtype=torch.float64,device=dev))
            if int(info)==0:break
        else:raise RuntimeError("Schur Cholesky failed")
    return L,d,jit
def qr_factor(Xb):
    """Preconditioner from QR of B=A(Y(x)Y), X=Y Y^T (Y=FP64 Cholesky factor of hi(X)); M~=B B^T. Loewner-close to M (rel. cond(X)*u), factored stably."""
    Bs=[]
    for k,(o,n) in enumerate(zip(offs,dims)):
        Y=torch.linalg.cholesky(Xb[k][0]);Ab=AH[:,o:o+n*n].reshape(R,n,n);Bs.append((Y.t()@Ab@Y).reshape(R,-1))
    B=torch.cat(Bs,1);d=1.0/B.norm(dim=1).clamp(min=1e-300);Bt=(B*d[:,None]).t().contiguous();del B,Bs
    Rm=torch.linalg.qr(Bt,mode="r")[1];return ("qr",Rm,d)
def pinv(F,r):
    if F[0]=="qr":
        _,Rm,d=F;u=(d*r).view(R,1);w=torch.linalg.solve_triangular(Rm.t(),u,upper=False);w=torch.linalg.solve_triangular(Rm,w,upper=True);return d*w.view(R)
    L,d,_=F;return (torch.cholesky_solve((r*d).view(R,1),L).view(R))*d
def gmres(op,z,k,tol):
    beta=float(z.norm());V=torch.zeros(R,k+1,dtype=torch.float64,device=dev);V[:,0]=z/beta;H=np.zeros((k+1,k));yk=None;j=0;est=1.0
    for j in range(k):
        w=op(V[:,j])
        for _ in range(2):
            h=V[:,:j+1].t()@w;w=w-V[:,:j+1]@h;H[:j+1,j]+=h.cpu().numpy()
        hn=float(w.norm());H[j+1,j]=hn
        e1=np.zeros(j+2);e1[0]=beta
        yk,res,_,_=np.linalg.lstsq(H[:j+2,:j+1],e1,rcond=None);est=float(np.linalg.norm(H[:j+2,:j+1]@yk-e1))/beta
        if est<tol or hn<1e-300:break
        V[:,j+1]=w/hn
    return V[:,:j+1]@T(yk),j+1,est
def ir_solve(Xb,F,rhs,tol,maxouter,gm):
    """GMRES-IR for M d = rhs (DD rhs). Returns DD d, info, final preconditioned relative residual."""
    d=dd.zeros(R,device=dev);r=rhs;z0=float(pinv(F,rhs[0]).norm());info=[];op=lambda v:pinv(F,Mv_dd(Xb,(v,torch.zeros_like(v)))[0])
    rel=1.0
    for it in range(maxouter):
        z=pinv(F,r[0]+r[1]);rel=float(z.norm())/max(z0,1e-300)
        if rel<tol:break
        delta,gi,est=gmres(op,z,gm,1e-6)
        d=dd.add(d,(delta,torch.zeros_like(delta)));Md=Mv_dd(Xb,d);r=dd.sub(rhs,Md);info.append((gi,round(math.log10(max(rel,1e-300)),1)))
    else:
        z=pinv(F,r[0]+r[1]);rel=float(z.norm())/max(z0,1e-300)
    return d,info,rel
def logdet_f(Zh):
    s=0.0
    for Z in Zh:
        L,info=torch.linalg.cholesky_ex(Z)
        if int(info)!=0:return None
        s+=2*float(torch.log(L.diagonal()).sum())
    return s
def phi_f(yh):
    v=torch.mv(AH.t(),yh);Zh=[t*I_b[k]-blocks((v,torch.zeros_like(v)))[k][0] for k in range(2)]
    ld=logdet_f(Zh);return None if ld is None else -float(torch.dot(b[0],yh))-ld
y=(torch.zeros(R,dtype=torch.float64,device=dev),torch.zeros(R,dtype=torch.float64,device=dev))
if a.resume:
    ck=np.load(a.resume);y=(T(ck["yh"]),T(ck["yl"]))
yh=y[0].clone();it=0
def f64_state(yh):
    v=torch.mv(AH.t(),yh);Zh=[t*I_b[k]-blocks((v,torch.zeros_like(v)))[k][0] for k in range(2)]
    Xh=[torch.linalg.inv(Z) for Z in Zh];Xh=[(X+X.t())/2 for X in Xh]
    g=torch.mv(AH,torch.cat([X.reshape(-1) for X in Xh]))-b[0]
    return Zh,Xh,g
phase1_done=a.resume is not None
while not phase1_done and it<a.p1:
    it+=1;Zh,Xh,g=f64_state(yh);M=schur64(Xh);F=chol_factor(M);dy=-pinv(F,g)
    lam2=-float(torch.dot(g,dy));f0=phi_f(yh);alpha=1.0
    while True:
        f1=phi_f(yh+alpha*dy)
        if f1 is not None and f1<=f0-1e-4*alpha*lam2:break
        alpha*=0.5
        if alpha<1e-10:break
    yh=yh+alpha*dy
    log(f"f64 it{it} |g|inf={float(g.abs().max()):.3e} lam2={lam2:.3e} alpha={alpha:.3g} phi={f0:.8e} jit={F[2]}")
    if alpha<1e-9 or lam2<a.lam1:phase1_done=True
y=(yh,torch.zeros_like(yh))
if a.f64only:sys.exit(0)
def save_ck():np.savez(f"{HERE}/data/ck_m{m}{a.tag}.npz",yh=y[0].cpu().numpy(),yl=y[1].cpu().numpy(),t=t)
# ---------------- phase 2 ----------------
hist=[];nit2=0
for it2 in range(1,a.maxit+1):
    nit2=it2;t1=time.time();t0=tic();Z=Zdd(y);Xb=[];rns=[]
    for k in range(2):
        X,rn=dd_inv(Z[k]);Xb.append(X);rns.append(rn)
    x=unblocks(Xb);g=dd.sub(Ax_dd(x),b);gn=float(g[0].abs().max());tadd("inverse+residual",t0)
    eigs=[torch.linalg.eigvalsh(X[0]) for X in Xb];emin=[float(e[0]) for e in eigs]
    if gn<a.tol:break
    t0=tic();F=qr_factor(Xb);tadd("schur_formation+QR_factorization",t0)
    t0=tic();dy,info,rel=ir_solve(Xb,F,dd.neg(g),a.ir_tol,a.ir_max,a.gm);tadd("refinement",t0)
    lam2=-float(torch.dot(g[0],dy[0]));lam=math.sqrt(max(lam2,0))
    hist.append(gn);log(f"dd it{it2} |g|inf={gn:.3e} lam2={lam2:.3e} Xmin={emin} IR(gmres_its,log10rel)={info} final_rel={rel:.1e} t={time.time()-t1:.1f}s")
    t0=tic();alpha=1.0;f0=phi_f(y[0])
    while True:
        ynew=dd.add(y,dd.mulf(dy,alpha));f1=phi_f(ynew[0])
        if f1 is not None and (lam<1e-2 or f1<=f0-1e-4*alpha*lam2):break
        alpha*=0.5
        if alpha<1e-12:raise RuntimeError("step failed")
    tadd("linesearch",t0);log(f"   alpha={alpha:.3g} phi={f0:.10e}")
    y=ynew
    if it2%5==0:save_ck()
save_ck()
# ---------------- final: primal Newton polish  X <- X - X mat(A^T dy) X, dy=M^-1 g  (DD residual) ----------------
log(f"barrier phase done: |g|inf={gn:.3e}, iterations={nit2}; starting primal polish")
bn=float(b[0].abs().max());polish=[]
for pit in range(8):
    x=unblocks(Xb);g=dd.sub(Ax_dd(x),b);gi=float(g[0].abs().max())
    log(f"polish {pit}: |Ax-b0|inf={gi:.3e} rel={gi/bn:.3e}");polish.append(gi/bn)
    if gi/bn<=a.final_tol:break
    t0=tic();F=qr_factor(Xb);tadd("schur_formation+QR_factorization",t0)
    t0=tic();dy,info,rel=ir_solve(Xb,F,g,1e-18,12,a.gm);tadd("refinement",t0);log(f"   IR {info} rel={rel:.1e}")
    W=blocks(Aty_dd(dy));Xb=[dd.sub(Xb[k],dd.matmul(dd.matmul(Xb[k],W[k]),Xb[k])) for k in range(2)]
    Xb=[dd.mulf(dd.add(X,(X[0].t().contiguous(),X[1].t().contiguous())),0.5) for X in Xb]
x=unblocks(Xb);g=dd.sub(Ax_dd(x),b);gi=float(g[0].abs().max())
eigX=[float(torch.linalg.eigvalsh(X[0])[0]) for X in Xb]
Ls=[torch.linalg.cholesky_ex(X[0]) for X in Xb];pd=all(int(i)==0 for _,i in Ls)
log(f"FINAL |Ax-b0|inf={gi:.3e} rel={gi/bn:.3e} min eig X={eigX} chol_ok={pd}")
if not pd or min(eigX)<=0:raise RuntimeError("final X not positive definite")
import mpmath
mpmath.mp.prec=200;grams=[]
for k,(X,n) in enumerate(zip(Xb,dims)):
    Xh=X[0].cpu().numpy();Xl=X[1].cpu().numpy();s=scales[k];G=[]
    for i in range(n):
        G.append([mpmath.nstr((mpmath.mpf(float(Xh[i,j]))+mpmath.mpf(float(Xl[i,j])))*norm/(s[i]*s[j]),36) for j in range(n)])
    grams.append(G)
np.savez(f"{HERE}/data/seed_m{m}_dd.npz",X0h=Xb[0][0].cpu().numpy(),X0l=Xb[0][1].cpu().numpy(),X1h=Xb[1][0].cpu().numpy(),X1l=Xb[1][1].cpu().numpy())
res={"order":m,"status":"mixed_precision_dual_barrier_newton","proved":False,"normalization":norm,"basis_scales":scales,"min_scaled_eigenvalues":eigX,
 "residual":gi,"residual_relative_to_b0_inf":gi/bn,"gram_matrices":grams,
 "solver":{"name":"tracks/F_mpsolver GMRES-IR + Ozaki DD","t":t,"margin":0,"barrier_iterations":nit2,"polish_rel_hist":polish,"timings_s":TM,"wall_s":time.time()-T0,"convention":"G = X*norm/outer(s,s); X in scaled units, A x = b0 (unshifted)"}}
json.dump(res,open(f"{HERE}/data/seed_m{m}.json","w"));open(f"{HERE}/data/READY_m{m}","w").write("ok\n")
log("wrote seed",TM)
