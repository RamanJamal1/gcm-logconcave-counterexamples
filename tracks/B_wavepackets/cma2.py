import sys,json,math,torch,numpy as np,cma,time
sys.path.insert(0,'.')
from wp import *
dev='cuda'
NF=int(sys.argv[1]); seed=int(sys.argv[2]); iters=int(sys.argv[3]); tag=sys.argv[4]
TS=np.array([0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.0,1.025,1.05])
def build(X):  # X [B,D] -> packets list ; D=3*NF + (NF-1)  : [y,z2,z3]*NF, logom for j>=1
    B=X.shape[0]; pk=[]
    for j in range(NF):
        om=torch.ones(B,1,device=dev,dtype=DT) if j==0 else torch.exp(X[:,3*NF+j-1:3*NF+j])
        y=X[:,3*j:3*j+1]; z2=X[:,3*j+1:3*j+2]; z3=X[:,3*j+2:3*j+3]
        a=torch.cat([0.07*torch.exp(y)/om**2, z2/(2*om)**15, z3/(3*om)**15],1)
        pk.append({'c':torch.zeros(B,1,device=dev,dtype=DT),'w':torch.full((B,1),1e12,device=dev,dtype=DT),'omega':om,'a':a.unsqueeze(1),'theta':torch.zeros(B,1,3,device=dev,dtype=DT)})
    return pk
def fit(X,npts=8001):
    B=X.shape[0]; pk=build(X)
    with torch.no_grad():
        L=200; n=20001
        x=torch.linspace(-L,L,n,device=dev,dtype=DT)[None].expand(B,n)
        lp,r0=logp_jet(x,'none',{'eps':torch.zeros(B,1,device=dev,dtype=DT)},pk)
        km=lp[...,2].max(-1).values*2; km=km.clamp_min(1e-3); rmin=r0.min(-1).values
        # S at TS*km
        kap=(km[:,None]*torch.tensor(TS,device=dev,dtype=DT)[None]).reshape(-1)
        pkb=[{k:v.repeat_interleave(len(TS),0) for k,v in P.items()} for P in pk]
        Ss=[];Gs=[]
        for c in range(0,B*len(TS),90):
            sl=slice(c,c+90)
            xg=make_grid(len(kap[sl]),13/kap[sl].sqrt(),npts)
            out,*_=evaluate(xg,'none',{'eps':kap[sl][:,None]},[{k:v[sl] for k,v in P.items()} for P in pkb],ms=(15,))
            Ss.append(out[15][0]);Gs.append(out[15][2]);del out,xg
        S=torch.cat(Ss).reshape(B,len(TS)); Gp=torch.cat(Gs).reshape(B,len(TS))
        neg=(S<0)
        idx=torch.where(neg,torch.tensor(TS,device=dev,dtype=DT)[None],torch.zeros_like(S)).max(-1).values
        r1=(S[:,12]/Gp[:,12])
        f=-idx+0.02*torch.asinh(r1/1e-10)/20
        f=torch.where(rmin<0.5,f+10,f); f=torch.where(torch.isfinite(f),f,torch.full_like(f,50))
    return f.cpu().numpy(),idx.cpu().numpy(),km.cpu().numpy()
rng=np.random.default_rng(seed); D=3*NF+NF-1
def samp():
    x=np.zeros(D)
    for j in range(NF): x[3*j]=rng.normal(-0.4,0.5); x[3*j+1]=rng.normal(0,1); x[3*j+2]=rng.normal(0,1)
    for j in range(1,NF): x[3*NF+j-1]=math.log(rng.uniform(1.05,2.6))
    return x
t0=time.time(); best=(9,None)
for s in range(3):
    P=np.array([samp() for _ in range(1024)]); F=np.concatenate([fit(torch.tensor(P[i:i+64],device=dev,dtype=DT))[0] for i in range(0,1024,64)])
    x0=P[np.argmin(F)]
    es=cma.CMAEvolutionStrategy(x0,0.3,{'popsize':64,'seed':seed*10+s,'verbose':-9})
    for it in range(iters):
        Xs=np.array(es.ask()); f,idx,km=fit(torch.tensor(Xs,device=dev,dtype=DT)); es.tell(Xs,f)
        j=int(f.argmin())
        if f[j]<best[0]: best=(float(f[j]),Xs[j].tolist(),float(idx[j]),float(km[j]))
    print(tag,'start',s,'best rho_hat',best[2],'f',best[0],f'{time.time()-t0:.0f}s',flush=True)
    json.dump({'NF':NF,'x':best[1],'rho15_grid':best[2],'kappa_min':best[3],'f':best[0]},open(f'data/cma_rho_{tag}.json','w'))
