import sys,json,math,torch,numpy as np
sys.path.insert(0,'.')
from wp import *
dev='cuda'
def packs_from(cfg):
    out=[]
    for P in cfg['packets']:
        K=len(P['a'])
        out.append({'c':torch.zeros(1,1,device=dev,dtype=DT)+P['c'],'w':torch.full((1,1),1e12,device=dev,dtype=DT),'omega':torch.full((1,1),P['omega'],device=dev,dtype=DT),
          'a':torch.tensor(P['a'],device=dev,dtype=DT).view(1,1,K),'theta':torch.tensor(P['theta'],device=dev,dtype=DT).view(1,1,K)})
    return out
def expand(pk,B):
    return [{k:v.expand(B,*v.shape[1:]).contiguous() for k,v in P.items()} for P in pk]
def kmin(pk,L=200,dx=0.01):
    n=int(2*L/dx)+1
    x=torch.linspace(-L,L,n,device=dev,dtype=DT)[None]
    lp,r0=logp_jet(x,'none',{'eps':torch.zeros(1,1,device=dev,dtype=DT)},pk)
    u2=lp[...,2]*2
    return float(u2.max()),float(r0.min())
def S(pk,kap,ms=(14,15,16),npts=20001):
    kap=torch.tensor(kap,device=dev,dtype=DT)
    B=len(kap)
    x=make_grid(B,13/kap.sqrt(),npts)
    out,*_=evaluate(x,'none',{'eps':kap[:,None]},expand(pk,B),ms=ms)
    return {m:out[m][0].cpu().numpy() for m in ms}
def rho(name,pk):
    km,rmin=kmin(pk); print(name,'kappa_min',km,'min r',rmin,flush=True)
    res={'kappa_min':km}
    for m in (14,15,16):
        pass
    ks=np.linspace(0.3,1.6,27)*km
    Ss={m:np.concatenate([S(pk,list(ks[i:i+9]))[m] for i in range(0,27,9)]) for m in (14,15,16)}
    for m in (14,15,16):
        neg=np.where(Ss[m]<0)[0]
        if len(neg)==0: print(' m',m,'S never negative on scan; signs',np.sign(Ss[m]).astype(int).tolist()); res[m]=None; continue
        i=neg.max(); 
        if i==len(ks)-1: print(' m',m,'negative at top of scan'); res[m]='>1.6';continue
        lo,hi=ks[i],ks[i+1]
        for _ in range(30):
            mid=(lo+hi)/2
            if S(pk,[mid],ms=(m,))[m][0]<0: lo=mid
            else: hi=mid
        res[m]=lo/km; print(' m',m,'kappa*',lo,'rho',lo/km,'signs',np.sign(Ss[m]).astype(int).tolist(),flush=True)
    return res
if __name__=='__main__':
    wit={'packets':[{'c':0,'omega':1.0,'a':[1/15,1/40000,-1/18e6],'theta':[0,0,0]}]}
    out={'witness':rho('witness',packs_from(wit))}
    for f in sys.argv[1:]:
        cfg=json.load(open(f)); out[f]=rho(f,packs_from(cfg))
    json.dump(out,open('data/kappa_rho.json','w'),indent=1)
