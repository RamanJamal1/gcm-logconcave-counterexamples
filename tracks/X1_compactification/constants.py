"""Rigorous Arb upper bounds C_k >= |kappa_k(X)| for X 1-strongly log-concave on R (k=3..16).
Proof chain (REPORT.md Thm 3): |mu_j| <= E|X-EX|^j <= M_j, M_2=1 (Brascamp-Lieb), M_j = E|Z-Z'|^j = 2^j Gamma((j+1)/2)/sqrt(pi) (Caffarelli + Jensen);
kappa_k = sum over set partitions of [k] without singletons of (-1)^(b-1)(b-1)! prod mu_|B|.
Also checks the Tweedie identity symbolically for a test prior. Gated by explicit raises."""
import sys, json
from flint import arb, ctx
from math import factorial
from fractions import Fraction as F
ctx.prec = 256
def M(j):
    if j == 2: return arb(1)
    return arb(2)**j * (arb(j+1)/2).gamma() / arb.pi().sqrt()
def parts(n, mx=None):
    mx = n if mx is None else mx
    if n == 0: yield []; return
    for f in range(min(n, mx), 1, -1):
        for r in parts(n-f, f): yield [f]+r
def nset(lam):
    k = sum(lam); c = factorial(k)
    for x in lam: c //= factorial(x)
    from collections import Counter
    for v in Counter(lam).values(): c //= factorial(v)
    return c
# self-test of moment-cumulant formula on exact central moments of Exp(1)-centered (kappa_k=(k-1)!)
import sympy as sp
x = sp.symbols('x')
mu = {j: sp.integrate((x-1)**j*sp.exp(-x), (x, 0, sp.oo)) for j in range(2, 17)}
for k in range(3, 17):
    kap = sum(nset(l)*(-1)**(len(l)-1)*factorial(len(l)-1)*sp.prod([mu[j] for j in l]) for l in parts(k))
    if sp.simplify(kap - factorial(k-1)) != 0: raise SystemExit(f'moment-cumulant self-test failed k={k}')
out = {}
for k in range(3, 17):
    C = sum(arb(nset(l)*factorial(len(l)-1)) * __import__('functools').reduce(lambda a, b: a*b, [M(j) for j in l], arb(1)) for l in parts(k))
    out[k] = C
    print(k, C.str(20, radius=True), 'upper', C.upper().str(12))
# Tweedie check: p = Gamma-like log-concave prior x^2 e^{-x} on x>0, s=1: (log q)^(k) = kappa_k(posterior) for k>=3, -1+Var for k=2
y = sp.symbols('y', real=True)
for yv in [sp.Rational(1, 3), 2]:
    post = lambda f: sp.Integral(f*x**2*sp.exp(-x)*sp.exp(-(x-y)**2/2), (x, 0, sp.oo))
    Z = post(1)
    logq = sp.log(Z)  # Z already includes exp(-(x-y)^2/2)
    K = lambda t: sp.log(sp.Integral(x**2*sp.exp(-x - x**2/2 + t*x), (x, 0, sp.oo)))
    for k in (2, 3, 4):
        lhs = sp.diff(logq, y, k).subs(y, yv).doit().evalf(30)
        t = sp.symbols('t'); rhs = (sp.diff(K(t), t, k).subs(t, yv).doit().evalf(30) - (1 if k == 2 else 0))
        if abs(lhs - rhs) > 1e-20: raise SystemExit(f'Tweedie check failed k={k} y={yv}: {lhs} {rhs}')
print('Tweedie and moment-cumulant checks passed')
json.dump({k: v.str(25, radius=True) for k, v in out.items()}, open('constants.json', 'w'), indent=1)
