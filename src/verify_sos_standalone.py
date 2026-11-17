"""Independent exact certificate verifier using only Python's standard library.

No SymPy, floating-point solver, generated integrand, or stored LDL is trusted.
Sparse polynomials and Schur complements use fractions.Fraction throughout.
"""
import ast,json,math,argparse,time
from fractions import Fraction as Q
from pathlib import Path
N=32
ZERO=(0,)*N

def add(a,b):
 c=a.copy()
 for k,v in b.items():
  c[k]=c.get(k,Q(0))+v
  if not c[k]:del c[k]
 return c

def scale(a,q):return {k:v*q for k,v in a.items() if v*q}
def mul(a,b):
 c={}
 for k,v in a.items():
  for l,w in b.items():
   e=tuple(x+y for x,y in zip(k,l));c[e]=c.get(e,Q(0))+v*w
 return {k:v for k,v in c.items() if v}
def const(v):return {} if not v else {ZERO:Q(v)}
def var(i):
 p=list(ZERO);p[i-1]=1;return {tuple(p):Q(1)}
def derivative(a,i):
 out={}
 for k,v in a.items():
  if k[i-1]:
   l=list(k);l[i-1]-=1;out[tuple(l)]=v*k[i-1]
 return out

def parse(text):
 def rec(n):
  if isinstance(n,ast.Constant) and isinstance(n.value,int):return const(n.value)
  if isinstance(n,ast.Name) and n.id.startswith('u'):return var(int(n.id[1:]))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return scale(rec(n.operand),-1)
  if isinstance(n,ast.BinOp):
   if isinstance(n.op,ast.Add):return add(rec(n.left),rec(n.right))
   if isinstance(n.op,ast.Sub):return add(rec(n.left),scale(rec(n.right),-1))
   if isinstance(n.op,ast.Mult):return mul(rec(n.left),rec(n.right))
   if isinstance(n.op,ast.Div):
    right=rec(n.right);assert set(right)=={ZERO};return scale(rec(n.left),1/right[ZERO])
   if isinstance(n.op,ast.Pow):
    assert isinstance(n.right,ast.Constant) and isinstance(n.right.value,int) and n.right.value>=0
    out=const(1)
    for _ in range(n.right.value):out=mul(out,rec(n.left))
    return out
  raise ValueError(ast.dump(n))
 return rec(ast.parse(text,mode='eval').body)

def total_derivative(a):
 out={}
 for i in range(1,N):out=add(out,mul(derivative(a,i),var(i+1)))
 return out

def entropy_polynomial(m):
 f=scale(mul(var(2),var(2)),Q(1,2))
 for _ in range(2,m):
  out={}
  for i in range(2,N-1):
   fi=derivative(f,i)
   for j in range(2,N-1):out=add(out,scale(mul(mul(derivative(fi,j),var(i+1)),var(j+1)),Q(-1,2)))
   for k in range(1,i):out=add(out,scale(mul(mul(fi,var(k+1)),var(i-k+1)),Q(math.comb(i,k),2)))
  f=out
 return scale(f,(-1)**m)

def monomials(weight,limit,i=1,exp=None):
 exp=list(ZERO) if exp is None else exp
 if i>limit:
  if weight==0:yield {tuple(exp):Q(1)}
  return
 for power in range(weight//i+1):
  exp[i-1]=power
  yield from monomials(weight-i*power,limit,i+1,exp.copy())

def positive_definite(matrix):
 a=[[Q(v) for v in row] for row in matrix];n=len(a)
 assert all(len(row)==n for row in a)
 assert all(a[i][j]==a[j][i] for i in range(n) for j in range(n))
 pivots=[]
 for k in range(n):
  pivot=a[k][k];assert pivot>0,('nonpositive pivot',k,str(pivot));pivots.append(pivot)
  for i in range(k+1,n):
   for j in range(i,n):
    a[i][j]-=a[i][k]*a[j][k]/pivot;a[j][i]=a[i][j]
 return pivots

def verify(path):
 start=time.time();d=json.loads(Path(path).read_text());m=d['order'];out={}
 assert len(d['bases'])==len(d['gram_matrices'])==2
 for block,(base,matrix) in enumerate(zip(d['bases'],d['gram_matrices'])):
  positive_definite(matrix);basis=[parse(v) for v in base];assert len(basis)==len(matrix)
  blockpoly={}
  for i,x in enumerate(basis):
   for j,y in enumerate(basis):blockpoly=add(blockpoly,scale(mul(x,y),Q(matrix[i][j])))
  out=add(out,blockpoly if block==0 else scale(mul(var(2),blockpoly),-1))
 if 'ibp_generators' in d:
  ps=[parse(v) for v in d['ibp_generators']]
  assert all(all(sum((i+1)*n for i,n in enumerate(e))==2*m-1 for e in p) for p in ps)
  identities=[add(total_derivative(p),mul(var(1),p)) for p in ps]
 else:identities=[add(total_derivative(p),mul(var(1),p)) for p in monomials(2*m-1,m-1)]
 assert len(d['ibp_polynomials'])==len(d['ibp_multipliers'])==len(identities)
 for expr,coefficient in zip(d['ibp_polynomials'],d['ibp_multipliers']):
  ident=parse(expr);assert ident in identities;identities.remove(ident);out=add(out,scale(ident,-Q(coefficient)))
 assert not identities
 target=entropy_polynomial(m)
 if 'target_subtraction' in d:target=add(target,scale(parse(d['target_subtraction']),-1))
 assert out==target,'polynomial identity failed'
 return {'order':m,'target_subtraction':d.get('target_subtraction','0'),'exact_identity':True,'positive_definite_grams':True,'ibp_identities_verified':True,'verifier':'standard-library Fraction sparse polynomials and exact Schur complements','seconds':time.time()-start}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('certificates',nargs='+');a=p.parse_args();results=[]
 for path in a.certificates:
  r=verify(path);results.append(r);print(json.dumps(r),flush=True)
 Path('certs/independent_sos_verification.json').write_text(json.dumps(results,indent=2)+'\n')
