import sys, json, time; sys.path.insert(0,'/tmp')
from kinks import *
from scipy.optimize import differential_evolution, minimize
n=int(sys.argv[1]); mtar=int(sys.argv[2]); seed=int(sys.argv[3]); tmax=float(sys.argv[4])
def unpack(v):
    gaps=v[:n-1]; s0=-v[n-1]; inc=v[n:2*n]
    xs=np.concatenate([[0.0],np.cumsum(gaps)]); xs-=xs.mean()
    ss=s0+np.concatenate([[0.0],np.cumsum(inc)])
    return xs,ss
def f(v,h=0.1):
    xs,ss=unpack(v)
    if ss[-1]<=0.03: return 1.0+(0.03-ss[-1])
    try: r=S_R(xs,ss,ms=[mtar],h=h,margin=11.0)[mtar][2]
    except Exception as e: return 1.0
    return r if np.isfinite(r) else 1.0
bounds=[(0.05,10)]*(n-1)+[(0.05,8)]+[(0.02,8)]*n
t0=time.time()
class Stop(Exception): pass
best=[1e9,None]
def cb(xk,convergence=None):
    val=f(xk)
    if val<best[0]: best[0]=val; best[1]=list(xk)
    print(n,mtar,seed,'t=%.0f'%(time.time()-t0),'best %.4e'%best[0],flush=True)
    return time.time()-t0>tmax
r=differential_evolution(f,bounds,seed=seed,popsize=12,maxiter=10**6,tol=0,callback=cb,polish=False,init='sobol')
v=r.x; xs,ss=unpack(v)
print('final',r.fun, xs.tolist(), ss.tolist())
json.dump({'n':n,'m':mtar,'R_h0.1':float(r.fun),'xs':xs.tolist(),'ss':ss.tolist(),'v':v.tolist()},open('/tmp/opt_%d_%d_%d.json'%(n,mtar,seed),'w'))
