import math,json,torch
from search import *
def rep(name,x,bp,pk,base='none',half=60,npts=40001):
    xx=make_grid(1,torch.tensor([half],device=dev,dtype=DT),npts)
    out,u2,r,d=evaluate(xx,base,bp,pk,ms=(14,15,16))
    dx=float(xx[0,1]-xx[0,0]); mu=float((xx*d).sum()*dx); var=float((((xx-mu)**2)*d).sum()*dx)
    row={'name':name,'var':var}
    for m in (14,15,16):
        S,A,Gp=[float(v) for v in out[m]]
        row[m]={'S':S,'R':S/A,'rho':S/Gp,'Shat':S*var**m/(math.factorial(m-1)/2)}
    print(json.dumps(row),flush=True); return row
t=lambda v:torch.full((1,1),v,device=dev,dtype=DT)
rows=[]
rows.append(rep('gauss_N01',0,{'eps':t(1.)},[],half=14))
pk=[{'c':t(0.),'w':t(1e6),'omega':t(1.),'a':torch.tensor([[[1/15,1/40000,-1/18e6]]],device=dev,dtype=DT),'theta':torch.zeros(1,1,3,device=dev,dtype=DT)}]
rows.append(rep('witness',0,{'eps':t(1/14)},pk,half=14*math.sqrt(14)))
kap=0.059;a=[0.0558,1.86e-5,-3.99e-8,2.65e-10,-2.85e-12]
xx=make_grid(1,torch.tensor([12/math.sqrt(kap)],device=dev,dtype=DT),40001)
X=jvar(xx);lp=-0.5*kap*jmul(X,X)
for k,ak in enumerate(a):lp=lp+ak*jcos_t(k+1,1.0,0.0,xx)
u={j:lp[...,j]*FACT[j] for j in range(1,N+1)}
d=torch.exp(lp[...,0]-lp[...,0].max());d=d/torch.trapezoid(d,xx,dim=-1)[...,None]
dx=float(xx[0,1]-xx[0,0]);var=float((xx**2*d).sum()*dx);row={'name':'baseline2_gauss_fourier5','var':var}
for m in (14,15,16):
    G=(-1)**m*INTEGRANDS[m](u);S=float(torch.trapezoid(G*d,xx,dim=-1));A=float(torch.trapezoid(INTEGRANDS[m](u).abs()*d,xx,dim=-1));Gp=float(torch.trapezoid(G.clamp_min(0)*d,xx,dim=-1))
    row[m]={'S':S,'R':S/A,'rho':S/Gp,'Shat':S*var**m/(math.factorial(m-1)/2)}
print(json.dumps(row));rows.append(row)
json.dump(rows,open('data/baselines.json','w'),indent=1)
