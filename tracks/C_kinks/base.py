import sys,json; sys.path.insert(0,'/tmp')
from ext import *
from math import factorial
K=20
def measures(x,h,Q,q,label,ms=range(12,19)):
    u=logderivs(Q,K); q=q/(q.sum()*h)
    var=(x**2*q).sum()*h-((x*q).sum()*h)**2
    out={}
    for m in ms:
        F=F_eval(m,u); G=(-1)**m*F*q
        S=G.sum()*h; A=np.abs(G).sum()*h; P=np.clip(G,0,None).sum()*h
        out[m]=dict(S=S,R=S/A,rho=S/P,Svar=S*var**m,gauss=factorial(m-1)/2)
    print(label,'Var=%.5f'%var)
    for m,o in out.items(): print('  m=%d S=%+.6e R=%+.3e rho=%+.3e S*Var^m=%.4e  (m-1)!/2=%.4e ratio=%.4f'%(m,o['S'],o['R'],o['rho'],o['Svar'],o['gauss'],o['Svar']/o['gauss']))
    return out
def gauss_series(x,x0s=0.0,c=1.0,K=K):
    # taylor coeffs at x of exp(-(x-x0s)^2/(2c)) scaled to value 1: g'=-(x-x0s)/c g
    g=np.zeros((K+1,)+x.shape); g[0]=np.exp(-(x-x0s)**2/(2*c)); d=x-x0s
    for j in range(K): g[j+1]=(-d*g[j]/c-(g[j-1]/c if j>=1 else 0))/(j+1)
    return g
def cauchy(a,b):
    c=np.zeros_like(a)
    for k in range(a.shape[0]):
        for j in range(k+1): c[k]+=a[j]*b[k-j]
    return c
res={}
# (a) Gaussian
x=np.arange(-14,14+1e-9,0.05); h=0.05
g=gauss_series(x); res['gauss']=measures(x,h,g,g[0],'Gaussian N(0,1)')
# S_m check at m=6,10,15 for Gaussian with v=1 and v=2.5
for v in [1.0,2.5]:
    xx=np.arange(-14*v**.5,14*v**.5,0.02*v**.5); hh=0.02*v**.5
    gg=gauss_series(xx,0,v); uu=logderivs(gg,K); qq=gg[0]/(gg[0].sum()*hh)
    for m in (6,10,15):
        S=(-1)**m*np.sum(F_eval(m,uu)*qq)*hh
        print('Gauss v=%g m=%d S=%.10e exact=%.10e'%(v,m,S,factorial(m-1)/(2*v**m)))
# (b) witness
xw=np.arange(-70,70+1e-9,0.05); hw=0.05
def cosser(x,k,K=K):
    c=np.zeros((K+1,)+x.shape)
    for j in range(K+1):
        c[j]=np.real(np.exp(1j*k*x)*(1j*k)**j)/factorial(j) if False else 0
    # d^j cos(kx) = k^j cos(kx + j pi/2)
    for j in range(K+1): c[j]=k**j*np.cos(k*x+j*np.pi/2)/factorial(j)
    return c
W=np.zeros((K+1,)+xw.shape); W[0]+=1
W+=cosser(xw,1)/15+cosser(xw,2)/40000-cosser(xw,3)/18000000
W[0]+=0  # constant 1 already counted at j=0
Gs=gauss_series(xw,0,14.0)
P=cauchy(Gs,W)
res['witness']=measures(xw,hw,P,P[0],'Witness t=0',ms=range(12,19))
# direct witness S_15,S_16
# (c) kink optima
d=json.load(open('/tmp/opt_5_15_1.json'))
for hh in (0.05,0.025):
    xs,ss=d['xs'],d['ss']; p_=setup(xs,ss); x,h_=grid_for(xs,ss,13.0,hh)
    Q,L=jets(x,xs,ss,K,p_); q=Q[0]*np.exp(L-L.max())
    res['n5_h%g'%hh]=measures(x,h_,Q,q,'n=5 optimum h=%g'%hh)
xs,ss=[0.0],[-1.0,1.0]
x,h_=grid_for(xs,ss,13.0,0.05); Q,L=jets(x,xs,ss,K,setup(xs,ss)); q=Q[0]*np.exp(L-L.max())
res['single_laplace']=measures(x,h_,Q,q,'single kink Laplace s=(-1,1)')
xs,ss=[-1.0,1.0],[-1,0,1]
d3=json.load(open('/tmp/opt_3_15_1.json'))
for nm,dd in (('n3',d3),):
    xs,ss=dd['xs'],dd['ss']; x,h_=grid_for(xs,ss,13.0,0.05); Q,L=jets(x,xs,ss,K,setup(xs,ss)); q=Q[0]*np.exp(L-L.max())
    res[nm]=measures(x,h_,Q,q,'n=3 optimum')
json.dump({k:{str(m):v for m,v in o.items()} for k,o in res.items()},open('/tmp/baselines.json','w'),default=float)
