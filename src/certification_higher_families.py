"""Entire/meromorphic density jets and proved real-axis tail majorants."""
import math
from flint import arb,acb

def logjets(q,absolute=False):
 u={1:q[1]}
 for n in range(2,13):
  rest=sum(math.comb(n-1,k-1)*u[k]*q[n-k] for k in range(1,n))
  u[n]=q[n]+rest if absolute else q[n]-rest
 return u

def hermite(n,z):
 a=1
 if n==0:return a
 b=z
 for j in range(1,n):a,b=b,z*b-j*a
 return b

def extra_family(c,exact,L):
 family=c['family']
 if family=='skew_gaussian_mixture':
  d=exact(c['separation_half']);weight=exact(c['weight'])
  if not (d>=0 and d<1 and weight>0 and weight<1):raise ValueError('invalid mixture parameters')
  bound=logjets({j:d**j for j in range(1,13)},True);bound[2]+=1
  tail=2*(-(L-d)**2/2).exp()/(L-d)
  def jets(x):
   p1=weight*(-(x-d)**2/2).exp();p2=(1-weight)*(-(x+d)**2/2).exp();p=p1+p2;r=p1/p
   u=logjets({j:d**j*(r+(-1)**j*(1-r)) for j in range(1,13)});u[2]-=1
   return p,u
  return jets,bound,1-d*d,tail,(2*arb.pi()).sqrt()
 if family!='rbf_curvature':raise ValueError('unsupported family')
 a=list(map(exact,c['weights']));s=list(map(exact,c['widths']));centers=list(map(exact,c['centers']));tilt=exact(c['tilt'])
 if not all(v>=0 for v in a) or not all(v>0 for v in s):raise ValueError('invalid RBF parameters')
 if not L>abs(tilt):raise ValueError('extent must exceed tilt')
 # |H_n(z)e^-z²/2| <= sum |coefficient_k| k^(k/2).
 # Each monomial bound follows by maximizing |z|^k e^-z²/2;
 # dropping e^-k/2 only enlarges it. This bound is uniform in the centers.
 def hbound(n):
  b=arb(0)
  for j in range(n//2+1):
   k=n-2*j;coeff=arb(math.factorial(n))/(2**j*math.factorial(j)*math.factorial(k))
   b+=coeff*(arb(k)**k).sqrt() if k else coeff
  return b
 bound={j:sum((aa/ss**(j-2)*hbound(j-2) for aa,ss in zip(a,s)),arb(0))+(1 if j==2 else 0) for j in range(2,13)}
 # Integrated RBF term is nonnegative: y erf(y/(sqrt2*s))>=0.
 # p_un <= exp(-x²/2+|tilt||x|), then complete the square.
 t=abs(tilt);tail=2*(t*t/2).exp()*(-(L-t)**2/2).exp()/(L-t)
 def jets(x):
  v=x*x/2+tilt*x;u={j:acb(-1 if j==2 else 0) for j in range(2,13)}
  for aa,ss,cc in zip(a,s,centers):
   z=(x-cc)/ss;e=(-z*z/2).exp()
   v+=aa*(ss*(arb.pi()/2).sqrt()*(x-cc)*(z/arb(2).sqrt()).erf()+ss*ss*e)
   for j in range(2,13):u[j]-=aa/ss**(j-2)*(-1)**(j-2)*hermite(j-2,z)*e
  return (-v).exp(),u
 return jets,bound,arb(1),tail,None
