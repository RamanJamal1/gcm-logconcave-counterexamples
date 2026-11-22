"""Independent exact certificate verifier using only Python's standard library.

No SymPy, floating-point solver, generated integrand, or stored LDL is trusted.
Sparse polynomials and Schur complements use fractions.Fraction throughout.

Every check raises CertificateError explicitly (no `assert`), so the verifier is
equally strict under `python -O`. The exit status is non-zero if any certificate
fails.
"""
import ast,json,math,argparse,sys,time
from fractions import Fraction as Q
from pathlib import Path
N=32
ZERO=(0,)*N

class CertificateError(Exception):
 """A certificate failed verification."""

def require(condition,message):
 """Raise CertificateError unless condition holds.

 `message` may be a zero-argument callable; it is only evaluated on failure, so
 building it never touches the (possibly enormous) exact numbers on success.
 """
 if not condition:raise CertificateError(message() if callable(message) else message)

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
 require(1<=i<=N,('variable index out of range',i))
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
    right=rec(n.right);require(set(right)=={ZERO},'division by a non-constant');return scale(rec(n.left),1/right[ZERO])
   if isinstance(n.op,ast.Pow):
    require(isinstance(n.right,ast.Constant) and isinstance(n.right.value,int) and n.right.value>=0,'exponent must be a non-negative integer')
    out=const(1)
    for _ in range(n.right.value):out=mul(out,rec(n.left))
    return out
  raise CertificateError('unsupported expression: '+ast.dump(n))
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

def sharp_subtraction(m):
 """The term 1/2 (m-1)! (-u2)^m that a sharp certificate must subtract."""
 e=list(ZERO);e[1]=m
 return {tuple(e):Q(math.factorial(m-1),2)*(-1)**m}

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
 require(n>0 and all(len(row)==n for row in a),'Gram matrix is not square')
 require(all(a[i][j]==a[j][i] for i in range(n) for j in range(n)),'Gram matrix is not symmetric')
 pivots=[]
 for k in range(n):
  pivot=a[k][k];require(pivot>0,lambda:('nonpositive pivot',k,f'{float(pivot):.6g}'));pivots.append(pivot)
  for i in range(k+1,n):
   for j in range(i,n):
    a[i][j]-=a[i][k]*a[j][k]/pivot;a[j][i]=a[i][j]
 return pivots

def verify(path):
 start=time.time();d=json.loads(Path(path).read_text());m=d['order'];out={}
 require(isinstance(m,int) and m>=2,'invalid order')
 require(len(d['bases'])==len(d['gram_matrices'])==2,'expected exactly two Gram blocks')
 for block,(base,matrix) in enumerate(zip(d['bases'],d['gram_matrices'])):
  positive_definite(matrix);basis=[parse(v) for v in base];require(len(basis)==len(matrix),('basis/Gram size mismatch in block',block))
  blockpoly={}
  for i,x in enumerate(basis):
   for j,y in enumerate(basis):blockpoly=add(blockpoly,scale(mul(x,y),Q(matrix[i][j])))
  out=add(out,blockpoly if block==0 else scale(mul(var(2),blockpoly),-1))
 if 'ibp_generators' in d:
  ps=[parse(v) for v in d['ibp_generators']]
  require(all(all(sum((i+1)*n for i,n in enumerate(e))==2*m-1 for e in p) for p in ps),'IBP generator of the wrong weight')
  identities=[add(total_derivative(p),mul(var(1),p)) for p in ps]
 else:identities=[add(total_derivative(p),mul(var(1),p)) for p in monomials(2*m-1,m-1)]
 require(len(d['ibp_polynomials'])==len(d['ibp_multipliers'])==len(identities),'IBP list lengths do not match the generators')
 for expr,coefficient in zip(d['ibp_polynomials'],d['ibp_multipliers']):
  ident=parse(expr);require(ident in identities,'listed IBP term is not an unused IBP identity');identities.remove(ident);out=add(out,scale(ident,-Q(coefficient)))
 require(not identities,'unused IBP identities remain')
 target=entropy_polynomial(m)
 if 'target_subtraction' in d:
  subtraction=parse(d['target_subtraction'])
  require(subtraction==sharp_subtraction(m),'target_subtraction is not exactly 1/2 (m-1)! (-u2)^m')
  target=add(target,scale(subtraction,-1))
 require(out==target,'polynomial identity failed')
 return {'order':m,'target_subtraction':d.get('target_subtraction','0'),'exact_identity':True,'positive_definite_grams':True,'ibp_identities_verified':True,'verifier':'standard-library Fraction sparse polynomials and exact Schur complements','seconds':time.time()-start}

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('certificates',nargs='+')
 p.add_argument('--output',help='also write the JSON results to this file')
 a=p.parse_args(argv);results=[];failed=0
 for path in a.certificates:
  try:r={'certificate':path,'ok':True,**verify(path)}
  except (CertificateError,KeyError,ValueError,SyntaxError,TypeError) as e:
   failed+=1;r={'certificate':path,'ok':False,'error':f'{type(e).__name__}: {e}'}
  results.append(r);print(json.dumps(r),flush=True)
 if a.output:Path(a.output).write_text(json.dumps(results,indent=2)+'\n',newline='\n')
 return 1 if failed else 0

if __name__=='__main__':
 sys.exit(main())
