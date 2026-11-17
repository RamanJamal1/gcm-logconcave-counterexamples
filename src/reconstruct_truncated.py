"""Recover an exact rational SOS and explicit IBP generators from a numerical SDP."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import argparse,json,time
from flint import fmpq,fmpq_mat
from pathlib import Path
import sympy as s
import numpy as np
import scipy.linalg as la
from sos_quotient import ROOT
from sos_truncated import setup as truncated_setup
from derive_integrands import u
from affine_pivots import select,ExactPivotSolve

def reconstruct(m,source):
 start=time.time();data=json.loads(Path(source).read_text());bases,target,keys,cols,layout,rows=truncated_setup(m,data['score_power_cap']);print('setup complete',m,time.time()-start,flush=True)
 scales=data['basis_scales'];normalization=data['normalization']
 top,bottom,chosen,tail,_,_=select(keys,cols,layout,scales);solver=ExactPivotSolve(keys,cols,top,bottom,chosen,tail)
 x=[s.Rational(str(data['gram_matrices'][block][i][j])) for block,i,j in layout];keyindex={k:i for i,k in enumerate(keys)};rhs=[fmpq(str(target.get(k,0))) for k in keys]
 for column,value in zip(cols,x):
  q=fmpq(int(value.p),int(value.q))
  for key,coefficient in column.items():rhs[keyindex[key]]-=fmpq(str(coefficient))*q
 correction=solver.solve_vector(rhs)
 for j,value in zip(chosen+tail,correction):x[j]+=s.Rational(str(value))
 residual_exact=[fmpq(str(target.get(k,0))) for k in keys]
 for column,value in zip(cols,x):
  q=fmpq(int(value.p),int(value.q))
  for key,coefficient in column.items():residual_exact[keyindex[key]]-=fmpq(str(coefficient))*q
 assert all(v==0 for v in residual_exact)
 print('exact square-system repair complete',m,time.time()-start,flush=True)
 grams=[s.zeros(len(base)) for base in bases]
 for v,(block,i,j) in zip(x,layout):grams[block][i,j]=grams[block][j,i]=v
 positive_counts=[]
 for Q in grams:
  assert Q==Q.T
  mat=fmpq_mat([[fmpq(int(v.p),int(v.q)) for v in row] for row in Q.tolist()])
  coefficients=list(mat.charpoly())
  # All eigenvalues are real by symmetry. Positivity of every coefficient
  # of det(xI+Q) excludes nonnegative roots, hence nonpositive eigenvalues.
  assert all(c*(-1)**(Q.rows-i)>0 for i,c in enumerate(coefficients)), 'exact Gram is not positive definite'
  positive_counts.append(len(coefficients));print('exact Gram checked',Q.rows,time.time()-start,flush=True)
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
 result={'order':m,'status':'exact_identity_and_positive_definite_grams_verified','scope':'integrated nonnegativity when entropy derivative and displayed IBP identities are valid','bases':[[str(mon(e)) for e in base] for base in bases],'gram_matrices':[[[str(v) for v in row] for row in Q.tolist()] for Q in grams],'ibp_generators':[str(p) for p,v in ps],'ibp_polynomials':[str(p) for p in ids],'ibp_multipliers':[str(v) for p,v in ps],'positive_characteristic_coefficient_counts':positive_counts,'identity':'(-1)^m F_m = v^T Q0 v + (-u2) w^T Q1 w - sum lambda_i (D P_i + u1 P_i)','source':str(source),'seconds':time.time()-start}
 (ROOT/f'certs/exact_truncated_m{m}_cap{data["score_power_cap"]}.json').write_text(json.dumps(result,indent=2)+'\n');print(m,result['status'],result['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);p.add_argument('source');a=p.parse_args();reconstruct(a.order,a.source)
