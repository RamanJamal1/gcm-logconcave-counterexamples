"""Recover an exact rational SOS and explicit IBP generators from a numerical SDP."""
import argparse,json,time
from pathlib import Path
import sympy as s
import numpy as np
import scipy.linalg as la
from sos_quotient import setup,ROOT
from derive_integrands import u

def reconstruct(m,source):
 start=time.time();data=json.loads(Path(source).read_text());bases,target,keys,cols,layout,rows=setup(m)
 scales=data['basis_scales'];normalization=data['normalization']
 factors=[s.Rational(normalization,scales[block][i]*scales[block][j]) for block,i,j in layout]
 A=s.Matrix([[d.get(k,0)*factor for d,factor in zip(cols,factors)] for k in keys]);b=s.Matrix([target.get(k,0) for k in keys])
 # Rank-revealing pivoting in scaled Gram coordinates controls rounding amplification.
 numerical=np.array(A,float);rn=np.maximum(np.linalg.norm(numerical,axis=1),1e-100)
 _,_,permutation=la.qr(numerical/rn[:,None],mode='economic',pivoting=True)
 permutation=list(map(int,permutation));permuted=A[:,permutation]
 R,piv=permuted.row_join(b).rref();assert len(cols) not in piv
 original=[s.Rational(str(data['gram_matrices'][block][i][j]))/factor for (block,i,j),factor in zip(layout,factors)]
 y=[original[j] for j in permutation];free=[j for j in range(len(y)) if j not in piv]
 for row,pivot in enumerate(piv):y[pivot]=R[row,-1]-sum(R[row,j]*y[j] for j in free)
 x=[s.Rational(0)]*len(y)
 for j,k in enumerate(permutation):x[k]=y[j]
 assert A*s.Matrix(x)==b
 x=[v*factor for v,factor in zip(x,factors)]
 grams=[s.zeros(len(base)) for base in bases]
 for v,(block,i,j) in zip(x,layout):grams[block][i,j]=grams[block][j,i]=v
 ldl=[]
 for Q in grams:
  L,D=Q.LDLdecomposition(hermitian=False);assert L*D*L.T==Q
  ds=[D[i,i] for i in range(D.rows)];assert all(v>0 for v in ds),'exact Gram is not positive definite';ldl.append(ds)
 # Gram minus target is decomposed into exact DP+u1 P identities by elimination.
 residual={k:-v for k,v in target.items()};generators={}
 def add(e,v):
  residual[e]=residual.get(e,0)+v
  if not residual[e]:del residual[e]
 for value,(block,i,j) in zip(x,layout):
  exp=[a+b for a,b in zip(bases[block][i],bases[block][j])];exp[1]+=block
  add(tuple(exp),value*(-1 if block else 1)*(1 if i==j else 2))
 while any(e[0] for e in residual):
  e=max((e for e in residual if e[0]),key=lambda e:e[0]);value=residual[e];p=list(e);p[0]-=1;P=tuple(p);generators[P]=generators.get(P,0)+value
  add(e,-value)
  for j,n in enumerate(p[:-1]):
   if n:
    f=p.copy();f[j]-=1;f[j+1]+=1;add(tuple(f),-value*n)
 assert not residual
 def mon(e):return s.prod(u[j+1]**power for j,power in enumerate(e))
 ps=[(mon(e),v) for e,v in sorted(generators.items()) if v]
 ids=[s.expand(u[1]*p+sum(s.diff(p,u[j])*u[j+1] for j in range(1,2*m))) for p,v in ps]
 result={'order':m,'status':'exact_identity_and_positive_definite_grams_verified','scope':'integrated nonnegativity when entropy derivative and displayed IBP identities are valid','bases':[[str(mon(e)) for e in base] for base in bases],'gram_matrices':[[[str(v) for v in row] for row in Q.tolist()] for Q in grams],'ibp_generators':[str(p) for p,v in ps],'ibp_polynomials':[str(p) for p in ids],'ibp_multipliers':[str(v) for p,v in ps],'ldl_positive_diagonal':[[str(v) for v in ds] for ds in ldl],'identity':'(-1)^m F_m = v^T Q0 v + (-u2) w^T Q1 w - sum lambda_i (D P_i + u1 P_i)','source':str(source),'seconds':time.time()-start}
 (ROOT/f'certs/exact_quotient_m{m}.json').write_text(json.dumps(result,indent=2)+'\n');print(m,result['status'],result['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);p.add_argument('source');a=p.parse_args();reconstruct(a.order,a.source)
