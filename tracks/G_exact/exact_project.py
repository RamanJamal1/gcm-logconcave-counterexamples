"""Exact rational projection of a numerical Gram seed onto the exact IBP-quotient constraint set.

Pipeline (all gating steps exact):
  1. round seed Gram entries to the dyadic grid 2^-K (exact integers X)
  2. exact residual R = target - C x  (python big ints)
  3. float64 iterative refinement in a weighted min-norm (W-orthogonal projection onto C x = target):
       dX = W^-1 C^T (C W^-1 C^T)^-1 R, rounded to the grid; exact residual recomputed each pass
       until |R| <= 2^-tol_bits (absolute)
  4. FLINT exact solve C_P z = R on a pivot subset of columns (Dixon/multimodular inside fmpz_mat.solve);
     z is tiny (|R| ~ 2^-tol) so pivot amplification is harmless.
  5. x = X/2^K + z with C x == target checked exactly over ALL keys.
Usage as module: project(m, seed_json_path, ...) -> dict
"""
import os,sys,json,time,math,pickle
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np,scipy.linalg as la,scipy.sparse as sp
from fractions import Fraction
from flint import fmpz,fmpq,fmpz_mat,fmpq_mat
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples')
sys.path.insert(0,ROOT+'/src')

def load_setup(m):
    d=pickle.load(open(f'{ROOT}/tracks/D_order14/data/setup_m{m}.pkl','rb'))
    return d['bases'],d['target'],d['keys'],d['cols'],d['layout']

def to_fraction(v):
    if isinstance(v,Fraction):return v
    if isinstance(v,int):return Fraction(v)
    return Fraction(str(v))

def parse_seed(seed,layout,dec_digits=None):
    """x[var] as exact Fraction from decimal strings/floats (symmetrised)."""
    G=seed['gram_matrices'];x=[]
    for b,i,j in layout:
        a=Fraction(str(G[b][i][j]));c=Fraction(str(G[b][j][i]))
        x.append((a+c)/2 if a!=c else a)
    return x

class System:
    def __init__(self,m,scales=None):
        self.m=m;self.bases,target,self.keys,self.cols,self.layout=load_setup(m)
        self.nk=len(self.keys);self.nv=len(self.cols);self.ix={k:i for i,k in enumerate(self.keys)}
        self.tgt=[to_fraction(target.get(k,0)) for k in self.keys]
        self.dt=1
        for t in self.tgt:self.dt=self.dt*t.denominator//math.gcd(self.dt,t.denominator)
        # sparse: per column list of (row,int)
        self.colsp=[[(self.ix[k],int(v)) for k,v in c.items()] for c in self.cols]
        self.rowsp=[[] for _ in range(self.nk)]
        for j,c in enumerate(self.colsp):
            for r,v in c:self.rowsp[r].append((j,v))
        self.scales=scales
    def residual_int(self,X,K):
        """R_i = dt*(2^K*target_i) - dt*sum C_ij X_j  as python ints (exact)."""
        R=[self.dt*(t.numerator*(1<<K)//t.denominator) for t in self.tgt]
        for j,c in enumerate(self.colsp):
            xj=X[j]
            if xj:
                for r,v in c:R[r]-=self.dt*v*xj
        return R
    def weights_inv(self,scales,norm=1):
        """winv_j = 1/(mult*(s_a s_b)^2): W-norm = Frobenius norm of scaled Gram."""
        w=[]
        for b,i,j in self.layout:
            s=scales[b][i]*scales[b][j];w.append(1.0/((1 if i==j else 2)*float(s)**2))
        return np.array(w)
    def float_matrix(self):
        R=[];Cc=[];V=[]
        for j,c in enumerate(self.colsp):
            for r,v in c:R.append(r);Cc.append(j);V.append(float(v))
        return sp.csr_matrix((V,(R,Cc)),shape=(self.nk,self.nv))

def pivots(sysm,scales):
    from affine_pivots import select
    top,bottom,chosen,tail,_,_=select(sysm.keys,sysm.cols,sysm.layout,scales)
    rows=top+bottom;cols=chosen+tail
    return rows,cols

def project(m,seed_path,kbits=256,tol_bits=160,arb_prec=700,out_path=None,max_iter=60,verbose=True,S=None):
    t0=time.time();log=lambda *a:print(f'[{time.time()-t0:7.1f}s]',*a,flush=True) if verbose else None
    seed=json.load(open(seed_path));scales=seed['basis_scales'];norm=seed['normalization']
    S=S or System(m);log('setup',S.nk,'keys',S.nv,'vars')
    xf=parse_seed(seed,S.layout)
    K=kbits;X=[round(v*(1<<K)) for v in xf]   # Fraction*int -> Fraction; round -> int (banker's, fine)
    R=S.residual_int(X,K);rel0=max(abs(r) for r in R)/(S.dt*(1<<K));log('initial abs residual',float(rel0))
    # high-precision (Arb) iterative refinement in a dyadic-weighted min-norm
    from flint import arb,arb_mat,ctx
    wf=S.weights_inv(scales);pj=[int(round(math.log2(w))) for w in wf];pmin=min(pj);Wi=[1<<(p-pmin) for p in pj]
    tM=time.time()
    Ca=fmpz_mat(S.nk,S.nv,[0]*(S.nk*S.nv))
    for j,c in enumerate(S.colsp):
        for r,v in c:Ca[r,j]=v
    CW=fmpz_mat(S.nk,S.nv,[0]*(S.nk*S.nv))
    for j,c in enumerate(S.colsp):
        for r,v in c:CW[r,j]=v*Wi[j]
    Mi=CW*Ca.transpose();log('exact integer M = C Wi C^T',S.nk,'x',S.nk,round(time.time()-tM,1),'s')
    ctx.prec=arb_prec;tI=time.time()
    Minv=arb_mat(Mi).inv();log('arb inverse at',arb_prec,'bits',round(time.time()-tI,1),'s; entry radius ok:',Minv[0,0].rad()<1e-30*abs(Minv[0,0].mid()) if Minv[0,0]!=0 else True)
    hist=[float(rel0)];it=0
    while it<max_iter:
        mx=max(abs(r) for r in R)
        if mx==0 or mx<=(S.dt*(1<<K))>>tol_bits:break
        e=mx.bit_length()
        rhs=arb_mat(S.nk,1,[arb(r)/(arb(2)**e) for r in R])
        y=Minv*rhs;sY=arb_prec-8
        Y=[]
        for i in range(S.nk):
            mid=y[i,0].mid();man,ex=mid.man_exp();man=int(man);ex=int(ex)
            sh=ex+sY      # Y_i = y_i*2^sY
            Y.append(man<<sh if sh>=0 else man>>(-sh))
        shift=e-sY
        for j,c in enumerate(S.colsp):
            tot=sum(v*Y[r] for r,v in c)
            if tot:
                num=Wi[j]*tot
                X[j]+=((num<<shift) if shift>=0 else (num+(1<<(-shift-1)))>>(-shift))//S.dt
        R=S.residual_int(X,K);it+=1;hist.append(float(max(abs(r) for r in R)/(S.dt*(1<<K))))
        if verbose:log('iter',it,'abs residual',hist[-1])
    # exact pivot solve of the tiny remainder
    rows,pcols=pivots(S,scales);n=len(rows)
    if n!=S.nk or len(pcols)!=n:raise RuntimeError('pivot set not square/full')
    rpos={r:i for i,r in enumerate(rows)}
    Mx=[[0]*n for _ in range(n)]
    for jj,j in enumerate(pcols):
        for r,v in S.colsp[j]:Mx[rpos[r]][jj]=v
    tsolve=time.time();Cp=fmpz_mat(Mx)
    rhs=fmpz_mat(n,1,[R[r] for r in rows])      # equation Cp z = R/(dt*2^K) ; solve for integer-scaled R
    z=Cp.solve(rhs)                              # fmpq_mat
    log('exact FLINT pivot solve',n,'x',n,round(time.time()-tsolve,1),'s')
    denom=1<<K
    xe=[Fraction(X[j],denom) for j in range(S.nv)]
    zmax=0.0
    for jj,j in enumerate(pcols):
        q=z[jj,0];f=Fraction(int(q.p),int(q.q))/(S.dt*denom);xe[j]+=f;zmax=max(zmax,abs(float(f)))
    # exact check over all keys
    res=[-t for t in S.tgt]
    for j,c in enumerate(S.colsp):
        if xe[j]:
            for r,v in c:res[r]+=v*xe[j]
    if any(res):raise RuntimeError('exact residual nonzero after projection')
    log('EXACT residual == 0 over all',S.nk,'keys')
    # weighted distance to seed
    wn=math.sqrt(sum(((1 if i==j else 2)*(float(xe[v]-xf[v])*scales[b][i]*scales[b][j]/norm)**2) for v,(b,i,j) in enumerate(S.layout)))
    maxden=max(len(str(v.denominator)) for v in xe)
    info={'m':m,'kbits':K,'refine_iters':it,'residual_history':hist,'pivot_correction_max_abs':zmax,'weighted_frob_shift_scaled':wn,'max_denominator_digits':maxden,'seconds':time.time()-t0}
    log(info)
    return xe,info,S,seed

def grams_from_x(S,xe):
    grams=[[[Fraction(0)]*len(b) for _ in range(len(b))] for b in S.bases]
    for v,(b,i,j) in zip(xe,S.layout):grams[b][i][j]=v;grams[b][j][i]=v
    return grams
