"""Recover an exact rational SOS and explicit IBP generators from a numerical SDP."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1"
import argparse,json,time
from flint import fmpq,fmpq_mat
from pathlib import Path
import sympy as s
import numpy as np
import scipy.linalg as la
from mckean_quotient import setup,ROOT
from derive_integrands import u

def reconstruct(m,source):
 start=time.time();data=json.loads(Path(source).read_text());bases,target,keys,cols,layout,rows=setup(m);print('setup complete',m,time.time()-start,flush=True)
 scales=data['basis_scales'];normalization=data['normalization']
 factors=[s.Rational(normalization,scales[block][i]*scales[block][j]) for block,i,j in layout]
 A=s.Matrix([[d.get(k,0) for d in cols] for k in keys]);b=s.Matrix([target.get(k,0) for k in keys])
 # Rank-revealing pivoting in scaled Gram coordinates controls rounding amplification.
 numerical=np.array(A,float)*np.array([float(f) for f in factors])[None,:];rn=np.maximum(np.linalg.norm(numerical,axis=1),1e-100)
 # Prefer singleton columns to retain a sparse exact affine solve.
 single={}
 for j,d in enumerate(cols):
  if len(d)==1:
   key=next(iter(d));row=keys.index(key);strength=abs(numerical[row,j])
   if row not in single or strength>single[row][0]:single[row]=(strength,j)
 selected=[single[row][1] for row in sorted(single)]
 remaining_cols=[j for j in range(len(cols)) if j not in set(selected)]
 remaining_rows=[i for i in range(len(keys)) if i not in single]
 _,_,tail=la.qr((numerical/rn[:,None])[remaining_rows][:,remaining_cols],mode='economic',pivoting=True)
 permutation=selected+[remaining_cols[int(j)] for j in tail]
 print('singleton pivots',len(selected),'remaining rows',len(remaining_rows),flush=True)
 permuted=A[:,permutation]
 R,piv=permuted.row_join(b).rref();assert len(cols) not in piv;print('exact affine solve complete',m,time.time()-start,flush=True)
 original=[s.Rational(str(data['gram_matrices'][block][i][j])) for block,i,j in layout]
 y=[original[j] for j in permutation];free=[j for j in range(len(y)) if j not in piv]
 for row,pivot in enumerate(piv):y[pivot]=R[row,-1]-sum(R[row,j]*y[j] for j in free)
 x=[s.Rational(0)]*len(y)
 for j,k in enumerate(permutation):x[k]=y[j]
 assert A*s.Matrix(x)==b
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
 result={'order':m,'target_subtraction':str(s.factorial(m-1)*(-u[2])**m/2),'status':'exact_identity_and_positive_definite_grams_verified','scope':'integrated nonnegativity when entropy derivative and displayed IBP identities are valid','bases':[[str(mon(e)) for e in base] for base in bases],'gram_matrices':[[[str(v) for v in row] for row in Q.tolist()] for Q in grams],'ibp_generators':[str(p) for p,v in ps],'ibp_polynomials':[str(p) for p in ids],'ibp_multipliers':[str(v) for p,v in ps],'positive_characteristic_coefficient_counts':positive_counts,'identity':'(-1)^m F_m - (m-1)!/2 (-u2)^m = v^T Q0 v + (-u2) w^T Q1 w - sum lambda_i (D P_i + u1 P_i)','source':str(source),'seconds':time.time()-start}
 (ROOT/f'certs/exact_mckean_quotient_m{m}.json').write_text(json.dumps(result,indent=2)+'\n');print(m,result['status'],result['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);p.add_argument('source');a=p.parse_args();reconstruct(a.order,a.source)
