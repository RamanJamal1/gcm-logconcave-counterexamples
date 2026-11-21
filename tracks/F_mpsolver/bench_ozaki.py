import os,sys,time,math,json
import numpy as np,torch
from fractions import Fraction
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import ozaki,dd
dev="cuda";out={}
def tm(f,rep=3):
    f();torch.cuda.synchronize();t=time.time()
    for _ in range(rep):f()
    torch.cuda.synchronize();return (time.time()-t)/rep
def to_dd(F):  # numpy object array of Fractions -> DD torch
    h=np.vectorize(float)(F);l=np.vectorize(lambda f:float(f-Fraction(float(f))))(F)
    return (torch.from_numpy(h).to(dev),torch.from_numpy(l).to(dev))
def relerr(C,Fx):
    h=C[0].cpu().numpy();l=C[1].cpu().numpy();worst=0.0;ref=max(abs(float(x)) for x in Fx.ravel())
    for i in range(Fx.shape[0]):
        for j in range(Fx.shape[1]):
            e=abs((Fraction(float(h[i,j]))+Fraction(float(l[i,j]))-Fx[i,j]))
            worst=max(worst,float(e))
    return worst/ref
# ---- accuracy on random matrices (mixed magnitudes, cond-like cancellation) ----
rng=np.random.default_rng(1)
def rfrac(shape,spread):
    mant=rng.standard_normal(shape);ex=rng.integers(-spread,spread+1,shape)
    return np.vectorize(lambda a,e:Fraction(a)*Fraction(2)**int(e)*(1+Fraction(int(rng.integers(1,2**50)),2**80)))(mant,ex)
for name,(m,k,n,spread) in {"rand_uniform":(40,160,40,0),"rand_spread8":(40,160,40,8)}.items():
    A=rfrac((m,k),spread);B=rfrac((k,n),spread)
    Fx=A.dot(B)  # exact object matmul
    Ad=to_dd(A);Bd=to_dd(B)
    r={}
    r["fp64_slices_relerr"]=relerr(dd.matmul(Ad,Bd),Fx);r["int8_slices_relerr"]=relerr(ozaki.i8_matmul(Ad,Bd),Fx)
    r["plain_fp64_relerr"]=relerr((torch.matmul(Ad[0],Bd[0]),torch.zeros(m,n,device=dev,dtype=torch.float64)),Fx)
    out[name]=r;print(name,r,flush=True)
# ---- accuracy on real problem: block of M = A diag(w) A^T for m13 rows, exact sparse reference ----
P=np.load(os.path.expanduser("~/gcm-logconcave-counterexamples/tracks/F_mpsolver/data/dd_m13.npz"))
R=len(P["bh"]);N=int(sum(int(d)**2 for d in P["dims"]))
AH=np.zeros((R,N));AL=np.zeros((R,N));AH[P["R"],P["C"]]=P["H"];AL[P["R"],P["C"]]=P["L"]
w=10.0**rng.uniform(-10,0,N)  # diagonal weights spanning 10 decades (like X (x) X)
wf=[Fraction(float(x)) for x in w]
rows=list(range(0,R,R//24))[:24]
ent={r:{int(c):Fraction(float(AH[r,c]))+Fraction(float(AL[r,c])) for c in np.nonzero(AH[r])[0]} for r in rows}
Fx=np.empty((len(rows),len(rows)),dtype=object)
for i,r in enumerate(rows):
    for j,s in enumerate(rows):
        Fx[i,j]=sum((v*wf[c]*ent[s][c] for c,v in ent[r].items() if c in ent[s]),Fraction(0))
Ah=torch.from_numpy(AH).to(dev);Al=torch.from_numpy(AL).to(dev)
Wd=(Ah[rows]*torch.from_numpy(w).to(dev),Al[rows]*torch.from_numpy(w).to(dev))
# w*A_row is NOT exactly DD (rounded) -> compare against the exactly-rounded operands: recompute reference with rounded W A
WAh=Wd[0];WAl=Wd[1]
WAf=[{int(c):Fraction(float(WAh[i,c]))+Fraction(float(WAl[i,c])) for c in np.nonzero(AH[r])[0]} for i,r in enumerate(rows)]
for i,r in enumerate(rows):
    for j,s in enumerate(rows):
        Fx[i,j]=sum((v*ent[s][c] for c,v in WAf[i].items() if c in ent[s]),Fraction(0))
At=(Ah[rows].t().contiguous(),Al[rows].t().contiguous())
r={}
r["fp64_slices_relerr"]=relerr(dd.matmul(Wd,At),Fx);r["int8_slices_relerr"]=relerr(ozaki.i8_matmul(Wd,At),Fx)
r["plain_fp64_relerr"]=relerr((torch.matmul(WAh,At[0]),torch.zeros(len(rows),len(rows),device=dev,dtype=torch.float64)),Fx)
out["m13_schur_block"]=r;print("m13_schur_block",r,flush=True)
# ---- timing: Schur-sized product R x N x R (m=13,14,15 sizes) ----
for mm,Rr,Nn in [(13,478,16130),(14,708,28226),(15,1039,49201)]:
    a=torch.randn(Rr,Nn,device=dev,dtype=torch.float64);b=torch.randn(Nn,Rr,device=dev,dtype=torch.float64)
    A=(a,a*1e-17);B=(b,b*1e-17)
    t={}
    t["torch_fp64"]=tm(lambda:a@b)
    t["torch_fp32"]=tm(lambda:a.float()@b.float())
    SA=ozaki.i8_split(*A);SB=ozaki.i8_split(B[0].t().contiguous(),B[1].t().contiguous())
    t["int8_gemm_stage_only"]=tm(lambda:ozaki.i8_mm_split(SA,SB))
    t["int8_split_A"]=tm(lambda:ozaki.i8_split(*A),1)
    t["int8_total"]=tm(lambda:ozaki.i8_matmul(A,B),1)
    SA6=dd.split_last(A[0],A[1],Nn);SB6=dd.split_last(B[0].t().contiguous(),B[1].t().contiguous(),Nn)
    t["fp64_slices_gemm_stage_only"]=tm(lambda:dd.mm_split(SA6,SB6),1);t["fp64_slices_L"]=len(SA6[1]);t["int8_L"]=len(SA[1])
    out[f"time_m{mm}_{Rr}x{Nn}x{Rr}"]=t;print(mm,t,flush=True)
    del a,b,A,B,SA,SB,SA6,SB6;torch.cuda.empty_cache()
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+"/data/bench_ozaki.json","w"),indent=1)
