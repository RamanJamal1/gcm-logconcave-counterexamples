"""Constrained small-kappa search (kappa<=CAP, |a_n|<=0.3), periodic surrogate. FP64 GPU."""
import math,sys,json,itertools,time
import torch
sys.path.insert(0,"src")
from periodic_cuda import evaluate
dev="cuda";dt=torch.float64
import os
CAP=float(os.environ.get('CAP','0.15'))
EPS=1e-4
def maxcurv(a,grid=4096):
    B,K=a.shape
    x=torch.linspace(-math.pi,math.pi,grid+1,dtype=dt,device=dev)[:-1]
    n=torch.arange(1,K+1,dtype=dt,device=dev)
    c=torch.cos(n[:,None]*x[None]);s=torch.sin(n[:,None]*x[None])
    r=1+a@c;r1=-(a*n)@s;r2=-(a*n*n)@c
    lc=(r*r2-r1**2)/r**2
    return lc.max(1).values,r.min(1).values
def kappa_of(a,grid=4096):
    m,rmin=maxcurv(a,grid);return (1+EPS)*m,rmin
def Qs(a,order,grid=2048):
    k,rmin=kappa_of(a,grid)
    S=evaluate(a,k,order,grid)
    return S/(0.5*math.factorial(order-1)*k**order),S,k,rmin
def P(a,order,grid=2048):
    return evaluate(a,torch.zeros(a.shape[0],dtype=dt,device=dev),order,grid)
WIT=[1/15,1/40000,-1/18000000]
def ablation():
    out={}
    def row(a,kap=1/14):
        a=torch.tensor([a],dtype=dt,device=dev);k=torch.tensor([kap],dtype=dt,device=dev)
        return [float(evaluate(a,k,m,4096)) for m in (14,15,16,17)],[float(P(a,n,4096)) for n in range(8,19)]
    print("witness",row(WIT));print("drop a3",row(WIT[:2]));print("drop a2,a3",row(WIT[:1]))
    print("n=8..18 P_n listed")
    for a3 in [-2e-6,-1e-6,-5e-7,-2e-7,-1e-7,-5.5e-8,-2e-8,-1e-8,0]:
        S,Pn=row([WIT[0],WIT[1],a3]);print("a3",a3,"S14-17",["%.3g"%v for v in S],"P12..16",["%.3g"%v for v in Pn[4:9]])
    for a4 in [-1e-9,-3e-10,-1e-10,0,1e-10,3e-10,1e-9]:
        S,Pn=row(WIT+[a4]);print("a4",a4,"S14-17",["%.3g"%v for v in S],"P12..16",["%.3g"%v for v in Pn[4:9]])

def build(sg,th):
    return sg*torch.exp(th.clamp(max=math.log(0.3)))
def seeds(B,K,g):
    th=torch.zeros(B,K,dtype=dt,device=dev)
    sg=torch.where(torch.rand(B,K,generator=g,device=dev)<0.5,-1.0,1.0).to(dt);sg[:,0]=1
    la1=torch.empty(B,dtype=dt,device=dev).uniform_(math.log(0.02),math.log(0.3),generator=g)
    th[:,0]=la1
    for n in range(2,K+1):
        lc=torch.empty(B,dtype=dt,device=dev).uniform_(-12,0.5,generator=g)
        th[:,n-1]=n*la1+lc
    nw=B//3
    w=torch.tensor(WIT,dtype=dt,device=dev)
    th[:nw,:3]=torch.log(w.abs())[None]+0.1*torch.randn(nw,3,generator=g,dtype=dt,device=dev)
    sg[:nw,:3]=torch.sign(w)[None]
    if K>3:
        th[:nw,3:]=torch.empty(nw,K-3,dtype=dt,device=dev).uniform_(-33,-18,generator=g)
    return sg,th
def run(m,K,B,steps,seed,lr=0.05):
    g=torch.Generator(device=dev);g.manual_seed(seed)
    sg,th=seeds(B,K,g);th.requires_grad_(True)
    opt=torch.optim.Adam([th],lr=lr)
    best=torch.full((B,),float("inf"),dtype=dt,device=dev);bth=th.detach().clone()
    for it in range(steps):
        a=build(sg,th);Q,S,k,rmin=Qs(a,m,512)
        bad=(k>CAP)|(rmin<0.5)|~torch.isfinite(Q)
        loss=torch.asinh(Q)+100*torch.relu(k-CAP)+100*torch.relu(0.5-rmin)
        loss=torch.where(torch.isfinite(loss),loss,torch.full_like(loss,50.))
        ok=(~bad)&(Q<best);best=torch.where(ok,Q.detach(),best);bth[ok]=th.detach()[ok]
        opt.zero_grad();loss.sum().backward();opt.step()
        if it%100==0:print(m,K,it,"bestQ %.4g"%best.min().item(),flush=True)
    return sg,bth,best
def final(m,K,sg,bth,best,top=5):
    idx=torch.argsort(best)[:top];res=[]
    for i in idx.tolist():
        a=build(sg[i:i+1],bth[i:i+1]);d={"a":[float(v) for v in a[0]]}
        for grid in (1024,2048,4096):
            Q,S,k,rmin=Qs(a,m,grid);d["Q_grid%d"%grid]=float(Q);d["S_grid%d"%grid]=float(S)
        d["kappa"]=float(k);d["rmin"]=float(rmin)
        d["Q15"]=float(Qs(a,15,2048)[0]);d["Q14"]=float(Qs(a,14,2048)[0]);d["S15"]=float(Qs(a,15,2048)[1])
        d["filter_scale"]=math.exp(-1/(2*d["kappa"]));res.append(d)
    return res
if __name__=="__main__":
    if sys.argv[1]=="ablation":ablation()
    else:
        m=int(sys.argv[1]);Ks=[int(v) for v in sys.argv[2].split(",")];B=int(sys.argv[3]);steps=int(sys.argv[4]);reps=int(sys.argv[5])
        out={}
        for K in Ks:
            allres=[]
            for r in range(reps):
                sg,bth,best=run(m,K,B,steps,1000*K+r);allres+=final(m,K,sg,bth,best)
                allres.sort(key=lambda d:d["Q_grid2048"]);allres=allres[:5]
                out[str(K)]=allres;json.dump(out,open("checkpoints/a1_constrained_m%d_cap%g_K%s.json"%(m,CAP,sys.argv[2].replace(",","")),"w"),indent=1)
            print("DONE K",K,allres[0],flush=True)

