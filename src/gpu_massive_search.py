"""Massive GPU CEM/ES search minimizing Q_m = S_m/(1/2 (m-1)! kappa^m) on the periodic surrogate."""
import sys,json,time,math,torch
from periodic_cuda import evaluate
import os
dev="cuda";SC=1e-12;FIX=float(os.environ.get("A1","0"))
def build(w,K,sine,dtype=torch.float64):
    a=SC*torch.sinh(w)
    if FIX:a=torch.cat([torch.full_like(a[:,:1],FIX),a[:,1:]],1)
    return a[:,:K],(a[:,K:] if sine else None)
def kappa_of(c,s,grid=2048,delta=1e-3):
    dt=c.dtype;B,K=c.shape
    th=torch.arange(grid,dtype=dt,device=dev)*2*math.pi/grid
    n=torch.arange(1,K+1,dtype=dt,device=dev)
    C=torch.cos(n[:,None]*th);Sn=torch.sin(n[:,None]*th)
    r=1+c@C;r1=-(c*n)@Sn;r2=-(c*n*n)@C
    if s is not None:
        r=r+s@Sn;r1=r1+(s*n)@C;r2=r2-(s*n*n)@Sn
    ok=(r.min(1).values>1e-3)
    rr=r.clamp_min(1e-3)
    cur=((r2*rr-r1*r1)/rr**2).max(1).values
    return (1+delta)*cur,ok
def Q(w,K,sine,order,grid=256,kgrid=2048,delta=1e-3):
    c,s=build(w,K,sine);kap,ok=kappa_of(c,s,kgrid,delta)
    ok=ok&(kap>1e-6)
    kk=kap.clamp_min(1e-6)
    S=evaluate(c,kk,order,grid,s)
    q=S/(0.5*math.factorial(order-1)*kk**order)
    q=torch.where(ok&torch.isfinite(q),q,torch.full_like(q,1e30))
    return q,kap
def run(name,K,sine,init,orders,gens,lam=65536,elite=0.02,sig0=1.0,seed=0,start=None,log=None):
    g=torch.Generator(device=dev);g.manual_seed(seed)
    D=K*(2 if sine else 1)
    if start is None:
        # random log-amplitude init
        a1=torch.exp(torch.empty(1,device=dev,dtype=torch.float64).uniform_(math.log(1e-3),math.log(0.4),generator=g))
        la=torch.empty(D,device=dev,dtype=torch.float64).uniform_(math.log(1e-12),math.log(0.4),generator=g)
        sg=torch.where(torch.rand(D,device=dev,generator=g)<.5,-1.,1.).double()
        mean=torch.asinh(sg*torch.exp(la)/SC)
        mean[0]=torch.asinh(a1[0]/SC)
        sig=torch.full((D,),sig0,device=dev,dtype=torch.float64)*2
    else:
        mean=torch.asinh(torch.tensor(start,dtype=torch.float64,device=dev)/SC);sig=torch.full((D,),sig0,device=dev,dtype=torch.float64)
    nel=max(8,int(lam*elite));best=(1e30,None,None,None);evals=0
    for gen in range(gens):
        order=orders[min(len(orders)-1,gen*len(orders)//gens)]
        w=mean+sig*torch.randn(lam,D,device=dev,dtype=torch.float64,generator=g)
        w[0]=mean
        q,kap=Q(w,K,sine,order);evals+=lam
        v,idx=torch.topk(q,nel,largest=False)
        el=w[idx];mean=el.mean(0);sig=(el.std(0)*1.05).clamp(1e-4,6)
        # track best for the current order
        if gen==gens-1 or gen*len(orders)//gens!=(gen+1)*len(orders)//gens: best=(1e30,None,None,None)
        if v[0].item()<best[0]:best=(v[0].item(),w[idx[0]].clone(),kap[idx[0]].item(),order)
    return best,evals
def refine(w,K,sine,order):
    out={}
    w=w[None]
    for gr in (256,1024,2048):
        q,kap=Q(w,K,sine,order,grid=gr);out[gr]=q.item()
    return out
if __name__=="__main__":
    fam=sys.argv[1];budget=float(sys.argv[2]);tag=sys.argv[3] if len(sys.argv)>3 else fam
    K,sine,mode=int(fam.split("_")[0][1:]),("s" in fam.split("_")[1]),fam.split("_")[2]
    t0=time.time();res={15:[],14:[]};tot=0;seed=int(time.time())%100000
    wit=[1/15,1/40000,-1/18000000]
    while time.time()-t0<budget:
        seed+=1
        for orders in ([15],[14]):
            if mode=="wit":
                st=wit+[1e-9]*(K-3);
                if sine:st=st+[0.0]*K
                st=[x*(1+0.0) for x in st]
                orders2=[16,16,15] if orders==[15] else [16,15,14]
                best,ev=run(fam,K,sine,None,orders2,150,seed=seed,start=[x if x!=0 else 1e-13 for x in st],sig0=0.5)
            else:
                best,ev=run(fam,K,sine,None,orders,200,seed=seed)
            tot+=ev
            if best[1] is None:continue
            m=orders[0];c=refine(best[1],K,sine,m)
            a=(SC*torch.sinh(best[1])).tolist()
            res[m].append({"Q":best[0],"Q_by_grid":c,"kappa":best[2],"amps":a,"K":K,"sine":sine,"seed":seed})
            print(fam,m,best[0],c,flush=True)
            for mm in (15,14):
                res[mm].sort(key=lambda d:d["Q"]);res[mm]=res[mm][:20]
            json.dump({"family":fam,"evals":tot,"best":res},open(f"checkpoints/ms_{tag}.json","w"),indent=1)
