"""Track C: exact jets of q = P_1 p for p ∝ exp(-V), V convex piecewise linear.
Knots xs (n), slopes ss (n+1, increasing). Segment i: V = s_i y + c_i.
T_i(x)=∫_seg e^{-V(y)}G(x-y)dy = e^{-c_i - s_i x + s_i^2/2}[Phi(x_{i+1}+s_i-x)-Phi(x_i+s_i-x)].
T_i' = -s_i T_i - (g_{i+1}-g_i), g_k(x)=e^{-V(x_k)}G(x-x_k);  q'=-sum s_i T_i.
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/gcm-logconcave-counterexamples/src'))
import numpy as np
from scipy.special import log_ndtr
from math import factorial
from generated_extended_integrands import INTEGRANDS

LOG2PI = 0.5*np.log(2*np.pi)

def setup(xs, ss):
    xs = np.asarray(xs, float); ss = np.asarray(ss, float)
    n = len(xs)
    if len(ss) != n+1: raise ValueError("need n+1 slopes")
    if np.any(np.diff(xs) <= 0) or np.any(np.diff(ss) <= 0): raise ValueError("order")
    if not (ss[0] < 0 < ss[-1]): raise ValueError("integrability")
    c = np.zeros(n+1)
    for i in range(n): c[i+1] = c[i] + (ss[i]-ss[i+1])*xs[i]
    Vk = ss[:n]*xs + c[:n]   # V(x_k) k=1..n (continuity)
    return xs, ss, c, Vk

def _logdiff_ndtr(zu, zl):
    """log(Phi(zu)-Phi(zl)), zl<zu, may be +-inf."""
    with np.errstate(all='ignore'):
        a = np.where(zl > 0, log_ndtr(-zl), log_ndtr(zu))
        b = np.where(zl > 0, log_ndtr(-zu), log_ndtr(zl))
        flip = zl > 0
        # straddle: direct
        direct = np.log(np.exp(log_ndtr(zu)) - np.exp(log_ndtr(zl)))
        # same-side: log(a)+log1p(-exp(b-a)); for zl<=0<... use zu<=0 branch
        a2 = np.where(zu <= 0, log_ndtr(zu), a)
        b2 = np.where(zu <= 0, log_ndtr(zl), b)
        same = a2 + np.log1p(-np.exp(b2 - a2))
        out = np.where((zl > 0) | (zu <= 0), same, direct)
    return out

def jets(x, xs, ss, K=16, params=None):
    """Return Q[k] Taylor coeffs of q at points x (scaled by exp(-Lref)), k=0..K, and Lref."""
    xs, ss, c, Vk = params if params is not None else setup(xs, ss)
    n = len(xs)
    x = np.asarray(x, float)
    bl = np.concatenate([[-np.inf], xs]); bu = np.concatenate([xs, [np.inf]])
    ell = [];  logD = []
    for i in range(n+1):
        zu = bu[i] + ss[i] - x; zl = bl[i] + ss[i] - x
        ld = _logdiff_ndtr(zu, zl)
        l = -c[i] - ss[i]*x + ss[i]**2/2
        ell.append(l); logD.append(ld)
    lg = [-Vk[k] - (x-xs[k])**2/2 - LOG2PI for k in range(n)]
    cand = [ell[i]+logD[i] for i in range(n+1)] + lg
    Lref = np.max(np.array(cand), axis=0)
    # knot gaussians' taylor coeffs
    gam = []
    for k in range(n):
        g = np.zeros((K+1,)+x.shape)
        g[0] = np.exp(lg[k]-Lref)
        d = x - xs[k]
        for j in range(K):
            g[j+1] = (-d*g[j] - (g[j-1] if j >= 1 else 0.0))/(j+1)
        gam.append(g)
    Q = np.zeros((K+1,)+x.shape)
    t = [np.zeros((K+1,)+x.shape) for _ in range(n+1)]
    for i in range(n+1):
        t[i][0] = np.exp(ell[i]+logD[i]-Lref)
        gu = gam[i] if i < n else 0.0     # g_{i+1} is knot index i (0-based) ; upper of segment i is knot i
        gl = gam[i-1] if i >= 1 else 0.0
        for j in range(K):
            gg = (gu[j] if i < n else 0.0) - (gl[j] if i >= 1 else 0.0)
            t[i][j+1] = (-ss[i]*t[i][j] - gg)/(j+1)
    for i in range(n+1): Q[0] += t[i][0]
    for j in range(K):
        Q[j+1] = -sum(ss[i]*t[i][j] for i in range(n+1))/(j+1)
    return Q, Lref

def logderivs(Q, K=16):
    """u_k = d^k log q for k=1..K from Taylor coeffs Q."""
    ph = np.zeros_like(Q)
    for k in range(1, K+1):
        acc = k*Q[k]
        for j in range(1, k):
            acc = acc - j*ph[j]*Q[k-j]
        ph[k] = acc/(k*Q[0])
    u = [None]*(K+1)
    for k in range(1, K+1): u[k] = ph[k]*factorial(k)
    return u

def grid_for(xs, ss, margin=14.0, h=0.05):
    lo = xs[0] - margin - abs(ss[0]); hi = xs[-1] + margin + ss[-1]
    return np.arange(lo, hi+h/2, h), h

def S_R(xs, ss, ms=range(2, 17), h=0.05, margin=14.0, params=None):
    p = params if params is not None else setup(xs, ss)
    x, h = grid_for(xs, ss, margin, h)
    Q, Lref = jets(x, xs, ss, 16, p)
    u = logderivs(Q)
    w = np.exp(Lref)*Q[0]      # q up to constant (e^{Lref} factor kept)
    # normalize weights: use q/∫q
    q = Q[0]*np.exp(Lref - Lref.max())
    q = q/(q.sum()*h)
    out = {}
    for m in ms:
        F = INTEGRANDS[m](u)
        S = (-1)**m*np.sum(F*q)*h
        A = np.sum(np.abs(F)*q)*h
        out[m] = (S, A, S/A)
    return out
