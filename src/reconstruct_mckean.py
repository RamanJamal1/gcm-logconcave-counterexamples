"""Exact rational reconstruction and independent polynomial/PSD checks."""
import argparse,json,time
from pathlib import Path
import sympy as s
from sos_probe import partitions
from derive_integrands import derive,u
ROOT=Path(__file__).resolve().parents[1]

def reconstruct(m):
 start=time.time();data=json.loads((ROOT/f'certs/sos_mckean_m{m}.json').read_text());symbols=[u[j] for j in range(1,m+1)]
 def mon(p):return s.prod(u[j] for j in p)
 def coeff(e):return dict(s.Poly(s.expand(e),*symbols).terms())
 bases=[[mon(p) for p in partitions(m,m) if max(p)>=3],[mon(p) for p in partitions(m-1,m-1) if max(p)>=3]]
 target=coeff(((-1)**m*derive(m)[m]-s.factorial(m-1)*(-u[2])**m/2));columns=[];approx=[];layout=[];ids=[]
 for b,(base,weight) in enumerate(zip(bases,[1,-u[2]])):
  for i in range(len(base)):
   for j in range(i,len(base)):
    columns.append(coeff(weight*base[i]*base[j]*(1 if i==j else 2)));approx.append(data['gram_matrices'][b][i][j]);layout.append((b,i,j))
 for j,parts in enumerate(partitions(2*m-1,m-1)):
  p=mon(parts);identity=u[1]*p+sum(s.diff(p,u[k])*u[k+1] for k in range(1,m));ids.append(identity)
  columns.append(coeff(-identity));approx.append(data['ibp_multipliers'][j]);layout.append(None)
 keys=sorted(set(target).union(*(set(d) for d in columns)));A=s.Matrix([[d.get(k,0) for d in columns] for k in keys]);b=s.Matrix([target.get(k,0) for k in keys])
 reduced,pivots=A.row_join(b).rref();assert len(columns) not in pivots,'inconsistent rational constraints'
 x=[s.Rational(str(round(v,12))) for v in approx];free=[j for j in range(len(x)) if j not in pivots]
 for row,pivot in enumerate(pivots):x[pivot]=reduced[row,-1]-sum(reduced[row,j]*x[j] for j in free)
 assert A*s.Matrix(x)==b
 grams=[s.zeros(len(base)) for base in bases];lams=[]
 for value,item in zip(x,layout):
  if item is None:lams.append(value)
  else:block,i,j=item;grams[block][i,j]=grams[block][j,i]=value
 # Exact rational LDL decomposition: positivity of every diagonal pivot is
 # a sufficient and independently checked proof of positive definiteness.
 pivots_out=[]
 for Q in grams:
  L,D=Q.LDLdecomposition(hermitian=False);assert L*D*L.T==Q
  ds=[D[i,i] for i in range(D.rows)];assert all(v>0 for v in ds),ds;pivots_out.append(ds)
 expr=sum((s.Matrix(base).T*Q*s.Matrix(base))[0]*weight for base,Q,weight in zip(bases,grams,[1,-u[2]]))-sum(lam*ident for lam,ident in zip(lams,ids))
 assert s.expand(expr-((-1)**m*derive(m)[m]-s.factorial(m-1)*(-u[2])**m/2))==0
 result={'order':m,'target_subtraction':str(s.factorial(m-1)*(-u[2])**m/2),'status':'exact_identity_and_positive_definite_grams_verified','scope':'integrated nonnegativity for log-concave densities for which the displayed integration-by-parts identities and derivative representation are valid','bases':[[str(v) for v in base] for base in bases],'gram_matrices':[[[str(v) for v in row] for row in Q.tolist()] for Q in grams],'ibp_polynomials':[str(v) for v in ids],'ibp_multipliers':[str(v) for v in lams],'ldl_positive_diagonal':[[str(v) for v in ds] for ds in pivots_out],'identity':'(-1)^m F_m - (m-1)!/2 (-u2)^m = v^T Q0 v + (-u2) w^T Q1 w - sum lambda_i (D P_i + u1 P_i)','seconds':time.time()-start}
 (ROOT/f'certs/exact_mckean_m{m}.json').write_text(json.dumps(result,indent=2)+'\n');print(m,result['status'],result['seconds'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[5,6,7,8]);a=p.parse_args()
 for m in a.orders:reconstruct(m)
