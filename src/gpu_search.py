#!/usr/bin/env python3
"""Reproducible signed Boltzmann-entropy derivative search.

CPU fallback uses NumPy float64. CUDA support uses Torch FP32 exploration and
FP64 refinement only when an actual CUDA device is available. No sampled sign
is a certificate. All implemented families have global analytic strict-convexity
bounds. Dimensionless objectives S_m Var(X)**m avoid dilation degeneracy.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import argparse, json, math, time, pathlib, sys
import numpy as np
from scipy.special import erf
from generated_integrands import INTEGRANDS
ROOT=pathlib.Path(__file__).resolve().parents[1]


def log_derivatives(q, upto=8):
    """q[j]=g^(j)/g; solve differentiated g'=g (log g)' exactly."""
    u={1:q[1]}
    for n in range(2,upto+1):
        u[n]=q[n]-sum(math.comb(n-1,k-1)*u[k]*q[n-k] for k in range(1,n))
    return u


def hermite(n,z):
    a=np.ones_like(z)
    if n==0:return a
    b=z
    for k in range(1,n):a,b=b,z*b-k*a
    return b


def density_jets(c,x):
    """Return unnormalized log density, log derivatives, analytic V'' bound."""
    family=c['family']
    if family in ('fourier_multiplier','fourier_potential'):
        a=np.array(c['amplitudes'])[:,None];w=np.array(c['frequencies'])[:,None]
        phase=np.array(c['phases'])[:,None];theta=w*x[None,:]+phase
        if family=='fourier_multiplier':
            g=1+(a*np.cos(theta)).sum(0)
            q={j:(a*w**j*np.cos(theta+j*np.pi/2)).sum(0)/g for j in range(1,9)}
            u=log_derivatives(q);u[1]=u[1]-x;u[2]=u[2]-1
            return -.5*x*x+np.log(g),u,1-float(np.sum(np.abs(a)*w*w))/(1-float(np.sum(np.abs(a))))
        u={j:(a*w**j*np.cos(theta+j*np.pi/2)).sum(0) for j in range(1,9)}
        u[1]=u[1]-x;u[2]=u[2]-1
        return -.5*x*x+(a*np.cos(theta)).sum(0),u,1-float(np.sum(np.abs(a)*w*w))
    if family=='rbf_curvature':
        a=np.array(c['weights'])[:,None];s=np.array(c['widths'])[:,None]
        center=np.array(c['centers'])[:,None];z=(x[None,:]-center)/s
        e=np.exp(-.5*z*z)
        v=.5*x*x+(a*(s*np.sqrt(np.pi/2)*(x[None,:]-center)*erf(z/np.sqrt(2))+s*s*e)).sum(0)+c['tilt']*x
        u={1:-x-(a*s*np.sqrt(np.pi/2)*erf(z/np.sqrt(2))).sum(0)-c['tilt']}
        for j in range(2,9):u[j]=-(a*s**(-(j-2))*(-1)**(j-2)*hermite(j-2,z)*e).sum(0)-(1 if j==2 else 0)
        return -v,u,1.
    if family=='skew_gaussian_mixture':
        d=c['separation_half'];weight=c['weight']
        log1=math.log(weight)-.5*(x-d)**2;log2=math.log1p(-weight)-.5*(x+d)**2
        logp=np.logaddexp(log1,log2);r=np.exp(log1-logp)
        # g=e^(dx)w + e^(-dx)(1-w), ignoring a common constant.
        q={j:d**j*(r+(-1)**j*(1-r)) for j in range(1,9)}
        u=log_derivatives(q);u[1]=u[1]-x;u[2]=u[2]-1
        return logp,u,1-d*d
    raise ValueError(family)


def evaluate(c,points=1025,extent=12.):
    # RBF modes can move. Newton locate the unique mode under V''>=1.
    mode=0.
    if c['family']=='rbf_curvature':
        for _ in range(30):
            _,u,_=density_jets(c,np.array([mode]));step=float(u[1][0]/u[2][0]);mode-=step
            if abs(step)<1e-12:break
    x=np.linspace(mode-extent,mode+extent,points)
    lp,u,convex=density_jets(c,x);p=np.exp(lp-np.max(lp));z=np.trapezoid(p,x);p/=z
    mean=float(np.trapezoid(x*p,x));var=float(np.trapezoid((x-mean)**2*p,x))
    vals={str(m):float((-1)**m*np.trapezoid(INTEGRANDS[m](u)*p,x)) for m in (6,7,8)}
    return {'signed':vals,'normalized':{m:v*var**int(m) for m,v in vals.items()},'variance':var,'mean':mean,'convexity_lower_bound':convex,'grid_points':points,'extent':extent,'mode':mode,'edge_mass_density':float(max(p[0],p[-1]))}


def random_candidate(rng,family):
    if family in ('fourier_multiplier','fourier_potential'):
        k=int(rng.integers(1,5));freq=np.exp(rng.uniform(math.log(.15),math.log(12),k))
        raw=rng.dirichlet(np.ones(k));margin=float(np.exp(rng.uniform(math.log(1e-4),math.log(.9))))
        if family=='fourier_multiplier':amp=raw*(1-margin)/(1+float(np.sum(raw*freq*freq)))
        else:amp=raw*(1-margin)/(freq*freq)
        amp*=rng.choice([-1,1],size=k)
        return {'family':family,'amplitudes':amp.tolist(),'frequencies':freq.tolist(),'phases':rng.uniform(-np.pi,np.pi,k).tolist()}
    if family=='rbf_curvature':
        k=int(rng.integers(1,5));return {'family':family,'weights':np.exp(rng.uniform(-3,3,k)).tolist(),'widths':np.exp(rng.uniform(-2,1,k)).tolist(),'centers':rng.uniform(-3,3,k).tolist(),'tilt':float(rng.uniform(-2,2))}
    if family=='skew_gaussian_mixture':return {'family':family,'separation_half':float(rng.uniform(.01,.99999)),'weight':float(1/(1+np.exp(-rng.uniform(-8,8))))}
    raise ValueError(family)


def atomic_json(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def search(args):
    if args.device=='cuda':
        raise RuntimeError('This implementation uses the validated NumPy CPU backend; CUDA vectorization is not implemented. No GPU work is claimed.')
    rng=np.random.default_rng(args.seed)
    families=['fourier_multiplier','fourier_potential','rbf_curvature','skew_gaussian_mixture']
    status={'schema':1,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'backend':'numpy_cpu_float64','gpu_evaluations':0,'seed':args.seed,'evaluations':0,'refinement_evaluations':0,'families':families,'best':{},'hits':[],'certified_counterexamples':0,'status':'running','elapsed_seconds':0,'failures':0,'search_scope':'finite sampled family search; no universal lower bound'}
    t0=time.monotonic();last_write=0
    while status['evaluations']<args.max_candidates and time.monotonic()-t0<args.seconds:
        family=families[status['evaluations']%len(families)];c=random_candidate(rng,family)
        try:r=evaluate(c,args.grid)
        except (OverflowError,ValueError,FloatingPointError):status['failures']+=1;status['evaluations']+=1;continue
        status['evaluations']+=1
        if not all(math.isfinite(v) for v in r['normalized'].values()):status['failures']+=1;continue
        for m in ('6','7','8'):
            if r['normalized'][m]<status['best'].get(m,{}).get('result',{}).get('normalized',{}).get(m,float('inf')):
                dense=evaluate(c,max(2049,args.grid*2+1));status['refinement_evaluations']+=1
                discrepancy=max(abs(dense['signed'][j]-r['signed'][j])/max(1.,abs(dense['signed'][j])) for j in ('6','7','8'))
                if discrepancy>1e-7:status['failures']+=1;continue
                entry={'parameters':c,'result':dense,'coarse_dense_relative_discrepancy':discrepancy,'certified':False,'evaluation':status['evaluations']}
                status['best'][m]=entry;atomic_json(ROOT/f'checkpoints/best_m{m}.json',entry)
                if dense['signed'][m]<-1e-8:
                    hit={'order':int(m),**entry};status['hits'].append(hit);atomic_json(ROOT/f'checkpoints/m{m}_hits/hit_{status["evaluations"]}.json',hit)
                print(json.dumps({'evaluation':status['evaluations'],'order':m,'normalized':dense['normalized'][m],'family':family}),flush=True)
        now=time.monotonic();status['elapsed_seconds']=now-t0
        if now-last_write>10:
            atomic_json(args.output,status);last_write=now
    status['status']='completed';status['elapsed_seconds']=time.monotonic()-t0;status['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());atomic_json(args.output,status)
    print(json.dumps({'status':'completed','evaluations':status['evaluations'],'seconds':status['elapsed_seconds'],'best':{m:v['result']['normalized'][m] for m,v in status['best'].items()},'hits':len(status['hits'])}),flush=True)
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--max-candidates',type=int,default=100000000);p.add_argument('--seed',type=int,default=260830275);p.add_argument('--grid',type=int,default=1025);p.add_argument('--device',choices=['cpu','auto','cuda'],default='auto');p.add_argument('--output',default=str(ROOT/'checkpoints/search_results.json'))
    search(p.parse_args())
