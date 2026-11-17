"""Strictly convex polynomial potentials: V'' = 1 + q(x)^2."""
import math,torch
from generated_higher_integrands import INTEGRANDS

def decode(raw):
 K=raw.shape[1]-1
 factors=torch.tensor([math.sqrt(math.factorial(j)) for j in range(K)],device=raw.device,dtype=raw.dtype)
 return 2*torch.tanh(raw[:,:K])/factors,3*torch.tanh(raw[:,-1])

def coefficients(raw):
 q,tilt=decode(raw);K=q.shape[1];s=[sum(q[:,i]*q[:,n-i] for i in range(K) if 0<=n-i<K) for n in range(2*K-1)]
 v=[torch.zeros_like(tilt),tilt]+[s[n]/((n+1)*(n+2)) for n in range(len(s))];v[2]=v[2]+.5
 return q,tilt,v

def horner(coeff,x):
 out=torch.zeros_like(x)
 for c in reversed(coeff):out=out*x+c[:,None]
 return out

def evaluate(raw,order=12,grid=513):
 q,tilt,v=coefficients(raw);degree=len(v)-1;vp=[j*v[j] for j in range(1,len(v))]
 with torch.no_grad():
  lo=-tilt.abs()[:,None]-1;hi=-lo
  for _ in range(35):
   mid=(lo+hi)/2;sign=horner(vp,mid);lo=torch.where(sign<0,mid,lo);hi=torch.where(sign<0,hi,mid)
  mode=(lo+hi)/2
 x=torch.linspace(-12,12,grid,device=raw.device,dtype=raw.dtype)[None,:]+mode
 lp=-horner(v,x);lp=lp-lp.max(1,keepdim=True).values;density=torch.exp(lp);density=density/torch.trapezoid(density,x,dim=1)[:,None]
 u={}
 for j in range(2,order+1):
  if j>degree:u[j]=0
  else:
   coeff=[-v[n]*math.factorial(n)/math.factorial(n-j) for n in range(j,len(v))];u[j]=horner(coeff,x)
   if raw.dtype==torch.float32:u[j]=torch.where(lp>-150,u[j],0)
 mean=torch.trapezoid(x*density,x,dim=1);variance=torch.trapezoid((x-mean[:,None])**2*density,x,dim=1)
 if raw.dtype==torch.float32:
  # Weighted homogeneity avoids overflow in large polynomial tail jets.
  scale=torch.ones(raw.shape[0],device=raw.device,dtype=raw.dtype)
  for j,vj in u.items():
   if isinstance(vj,torch.Tensor):scale=torch.maximum(scale,vj.abs().max(1).values**(2/j))
  scaled={j:vj/scale[:,None]**(j/2) if isinstance(vj,torch.Tensor) else vj for j,vj in u.items()}
  integral=(-1)**order*torch.trapezoid(INTEGRANDS[order](scaled)*density,x,dim=1)
  logabs=integral.abs().log();sign=integral.sign()
  value=sign*torch.exp(logabs+order*scale.log())
  normalized=sign*torch.exp(logabs+order*(scale*variance).log())
 else:
  value=(-1)**order*torch.trapezoid(INTEGRANDS[order](u)*density,x,dim=1);normalized=value*variance**order
 return normalized,value,variance

def refine(raw,order,steps=8,grid=2049):
 x=raw.detach().double().clone().requires_grad_(True);opt=torch.optim.LBFGS([x],max_iter=steps,line_search_fn='strong_wolfe')
 def closure():
  opt.zero_grad();value=evaluate(x,order,grid)[0];loss=torch.asinh(value/math.factorial(order-1)).sum();loss.backward();return loss
 opt.step(closure)
 with torch.no_grad():values=evaluate(x,order,grid)
 return x.detach(),tuple(v.detach() for v in values)
