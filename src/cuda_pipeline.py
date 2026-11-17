"""Batched CUDA quadrature and differentiable, globally convex parameter maps."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import math
import torch
from generated_integrands import INTEGRANDS
torch.set_num_threads(2)
FAMILIES=('fourier_multiplier','fourier_potential','rbf_curvature','skew_gaussian_mixture')

def decode(raw,family):
    # raw shape B x 4 x K; bounds prevent quadrature aliasing at initial grid.
    freq=.15+7.85*torch.sigmoid(raw[:,1])
    phase=math.pi*torch.tanh(raw[:,2])
    strength=.999*torch.sigmoid(raw[:,3,:1])
    a=torch.tanh(raw[:,0])
    if family=='fourier_multiplier':
        a=a*strength/(1+(a.abs()*(1+freq**2)).sum(-1,keepdim=True))
        return {'amplitudes':a,'frequencies':freq,'phases':phase}
    if family=='fourier_potential':
        a=a*strength/(1+(a.abs()*freq**2).sum(-1,keepdim=True))
        return {'amplitudes':a,'frequencies':freq,'phases':phase}
    if family=='rbf_curvature':
        return {'weights':torch.nn.functional.softplus(raw[:,0]),'widths':.3+2.7*torch.sigmoid(raw[:,1]),'centers':3*torch.tanh(raw[:,2]),'tilt':2*torch.tanh(raw[:,3,0])}
    return {'separation_half':.999*torch.sigmoid(raw[:,0,0]),'weight':.001+.998*torch.sigmoid(raw[:,1,0])}

def log_jets(q):
    u={1:q[1]}
    for n in range(2,9):u[n]=q[n]-sum(math.comb(n-1,k-1)*u[k]*q[n-k] for k in range(1,n))
    return u

def hermite(n,z):
    a=torch.ones_like(z)
    if n==0:return a
    b=z
    for j in range(1,n):a,b=b,z*b-j*a
    return b

def evaluate(raw,family,grid=513,extent=12.):
    p=decode(raw,family);base=torch.linspace(-extent,extent,grid,device=raw.device,dtype=raw.dtype)
    x=base[None,:].expand(raw.shape[0],-1)
    if family.startswith('fourier'):
        a=p['amplitudes'][...,None];w=p['frequencies'][...,None];phase=p['phases'][...,None];theta=w*x[:,None,:]+phase
        # Exact four-cycle avoids trig phase-shift rounding in high derivatives.
        co,si=torch.cos(theta),torch.sin(theta);cycle=(co,-si,-co,si)
        jets={j:(a*w**j*cycle[j%4]).sum(1) for j in range(9)}
        if family=='fourier_multiplier':
            g=1+jets[0];u=log_jets({j:jets[j]/g for j in range(1,9)})
            lp=-x*x/2+torch.log(g)
            convex=1-(p['amplitudes'].abs()*p['frequencies']**2).sum(1)/(1-p['amplitudes'].abs().sum(1))
        else:u={j:jets[j] for j in range(1,9)};lp=-x*x/2+jets[0];convex=1-(p['amplitudes'].abs()*p['frequencies']**2).sum(1)
        u[1]=u[1]-x;u[2]=u[2]-1
    elif family=='rbf_curvature':
        a=p['weights'][...,None];s=p['widths'][...,None];c=p['centers'][...,None];tilt=p['tilt'][:,None]
        # Differentiable bisection is only used to center the integration window;
        # detached center avoids artificial gradients from discrete branching.
        with torch.no_grad():
            bound=tilt.abs()+(a*s*math.sqrt(math.pi/2)).sum(1)+1;lo=-bound;hi=bound
            for _ in range(40):
                mid=(lo+hi)/2;vprime=mid+tilt+(a*s*math.sqrt(math.pi/2)*torch.erf((mid[:,None,:]-c)/s/math.sqrt(2))).sum(1)
                lo=torch.where(vprime<0,mid,lo);hi=torch.where(vprime<0,hi,mid)
            mode=(lo+hi)/2
        x=x+mode;z=(x[:,None,:]-c)/s;e=torch.exp(-z*z/2)
        lp=-x*x/2-tilt*x-(a*(s*math.sqrt(math.pi/2)*(x[:,None,:]-c)*torch.erf(z/math.sqrt(2))+s*s*e)).sum(1)
        u={j:-(a*s**(-(j-2))*(-1)**(j-2)*hermite(j-2,z)*e).sum(1)-(1 if j==2 else 0) for j in range(2,9)};convex=torch.ones(raw.shape[0],device=raw.device,dtype=raw.dtype)
    else:
        d=p['separation_half'][:,None];w=p['weight'][:,None]
        l1=w.log()-(x-d)**2/2;l2=torch.log1p(-w)-(x+d)**2/2;lp=torch.logaddexp(l1,l2);r=torch.exp(l1-lp)
        u=log_jets({j:d**j*(r+(-1)**j*(1-r)) for j in range(1,9)});u[1]=u[1]-x;u[2]=u[2]-1;convex=1-d[:,0]**2
    density=torch.exp(lp-lp.max(1,keepdim=True).values);density=density/torch.trapezoid(density,x,dim=1)[:,None]
    mean=torch.trapezoid(x*density,x,dim=1);var=torch.trapezoid((x-mean[:,None])**2*density,x,dim=1)
    signed=torch.stack([(-1)**m*torch.trapezoid(INTEGRANDS[m](u)*density,x,dim=1) for m in (6,7,8)],1)
    normalized=signed*var[:,None]**torch.tensor([6,7,8],device=raw.device)
    return normalized,signed,convex

def parameters(raw,family):
    d=decode(raw[None,:],family)
    return {'family':family,**{k:v[0].detach().cpu().tolist() for k,v in d.items()}}

def refine(raw,family,order,steps=8,grid=2049):
    x=raw.detach().to(dtype=torch.float64).clone().requires_grad_(True)
    optimizer=torch.optim.LBFGS([x],max_iter=steps,lr=.3,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();v=evaluate(x,family,grid)[0][:,order-6]
        loss=torch.asinh(v/math.factorial(order-1)).sum()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite refinement')
        loss.backward();return loss
    optimizer.step(closure)
    with torch.no_grad():out=evaluate(x,family,grid)
    return x.detach(),tuple(y.detach() for y in out)
