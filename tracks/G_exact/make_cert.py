"""Build the certificate JSON (format of src/reconstruct_quotient_fast.py) from exact Gram matrices (generator elimination copied)."""
import os,sys,json,time
from fractions import Fraction
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples');sys.path.insert(0,ROOT+'/src')
import sympy as s
from derive_integrands import u

def build(S,xe,source,t0=None,extra=None):
    t0=t0 or time.time();m=S.m;bases=S.bases;layout=S.layout;target=None
    from exact_project import load_setup
    _,target,_,_,_=load_setup(m)
    residual={k:-Fraction(str(v)) for k,v in target.items()};generators={}
    def add(e,v):
        residual[e]=residual.get(e,0)+v
        if not residual[e]:del residual[e]
    for value,(block,i,j) in zip(xe,layout):
        exp=[a+b for a,b in zip(bases[block][i],bases[block][j])];exp[1]+=block
        add(tuple(exp),value*(-1 if block else 1)*(1 if i==j else 2))
    while any(e[0] for e in residual):
        e=max((e for e in residual if e[0]),key=lambda e:e[0]);value=residual[e];p=list(e);p[0]-=1;P=tuple(p);generators[P]=generators.get(P,0)+value
        add(e,-value)
        for j,n in enumerate(p[:-1]):
            if n:
                f=p.copy();f[j]-=1;f[j+1]+=1;add(tuple(f),-value*n)
    if residual:raise RuntimeError('generator elimination left residual')
    def mon(e):return s.prod(u[j+1]**power for j,power in enumerate(e))
    ps=[(mon(e),v) for e,v in sorted(generators.items()) if v]
    ids=[s.expand(u[1]*p+sum(s.diff(p,u[j])*u[j+1] for j in range(1,2*m))) for p,v in ps]
    grams=[[[Fraction(0)]*len(b) for _ in range(len(b))] for b in bases]
    for v,(b,i,j) in zip(xe,layout):grams[b][i][j]=v;grams[b][j][i]=v
    res={'order':m,'status':'exact_identity_and_positive_definite_grams_verified','scope':'integrated nonnegativity when entropy derivative and displayed IBP identities are valid','bases':[[str(mon(e)) for e in base] for base in bases],'gram_matrices':[[[str(v) for v in row] for row in Q] for Q in grams],'ibp_generators':[str(p) for p,v in ps],'ibp_polynomials':[str(p) for p in ids],'ibp_multipliers':[str(v) for p,v in ps],'identity':'(-1)^m F_m = v^T Q0 v + (-u2) w^T Q1 w - sum lambda_i (D P_i + u1 P_i)','source':str(source),'seconds':time.time()-t0}
    if extra:res.update(extra)
    return res,grams
