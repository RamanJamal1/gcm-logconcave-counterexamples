"""Track A: complex-time singularities of F(t)=H(P_t p) for p ~ exp(-kappa x^2/2) R(x), periodic surrogate.
F(t) = -1/2 log(1+kappa t) + h(tau) + const, tau=t/(1+kappa t), h=<R log R>, R=1+sum a_n e^{-n^2 tau/2} cos(n x).
All numbers are NUMERICAL (mpmath, dps>=50); nothing here is a proof."""
import mpmath as mp, math, json, numpy as np
from math import comb

def _m(x): return mp.mpf(x)
CONFIGS = {
 'witness3': dict(a=[_m(1)/15, _m(1)/40000, -_m(1)/18000000], kappa=_m(1)/14),
 'two_mode': dict(a=[_m(1)/15, _m(1)/40000], kappa=_m(1)/14),
 'R15opt5': dict(a=[_m(x) for x in [0.05581601564460076, 1.8605329413968507e-05, -3.985583229780961e-08, 2.6536911012714195e-10, -2.852471396848522e-12]], kappa=_m(0.059041146116752596)),
 'zou32': dict(a=[_m(1)/256, _m(2)**-32], kappa=_m(1)/32),
}

def h_coeffs(a, N, grid=128, dps=60):
    """Exact Taylor coefficients h_0..h_N of h(tau)=<R log R> (trapezoid on grid; aliasing ~ e^{-grid*Im x0}, checked by grid doubling)."""
    mp.mp.dps = dps
    K = len(a); acc = [mp.mpf(0)]*(N+1)
    fact = [mp.factorial(k) for k in range(N+1)]
    for j in range(grid//2+1):
        w = (1 if j in (0, grid//2) else 2)
        x = 2*mp.pi*j/grid
        c = [mp.cos((k+1)*x) for k in range(K)]
        R = [sum(a[m]*c[m]*(-mp.mpf((m+1)**2)/2)**k/fact[k] for m in range(K)) for k in range(N+1)]
        R[0] += 1
        L = [mp.log(R[0])]
        for n in range(1, N+1):
            L.append((n*R[n] - sum(k*L[k]*R[n-k] for k in range(1, n)))/(n*R[0]))
        for n in range(N+1):
            acc[n] += w*sum(R[k]*L[n-k] for k in range(n+1))
    return [v/grid for v in acc]

def S_from_h(h, kappa, m):
    """periodic-surrogate S_m (repo formula)."""
    return mp.factorial(m)*sum(comb(m-1, n-1)*kappa**(m-n)*h[n] for n in range(1, m+1)) + mp.factorial(m-1)*kappa**m/2

def Rfun(a, tau, x):
    return 1 + sum(an*mp.e**(-(n+1)**2*tau/2)*mp.cos((n+1)*x) for n, an in enumerate(a))

def newton_sing(a, tau0, x0, maxit=60):
    """Solve R=R_x=0 in (tau,x) complex (double zero of R). Returns (tau,x) or None."""
    A = np.array([float(v) for v in a]); n = np.arange(1, len(A)+1)
    tau, x = complex(tau0), complex(x0)
    for _ in range(maxit):
        E = A*np.exp(-n**2*tau/2)
        cs, sn = np.cos(n*x), np.sin(n*x)
        R = 1+np.sum(E*cs); Rx = -np.sum(n*E*sn)
        Rt = -np.sum(n**2/2*E*cs); Rxx = -np.sum(n**2*E*cs); Rxt = np.sum(n**3/2*E*sn)
        J = np.array([[Rt, Rx], [Rxt, Rxx]]); f = np.array([R, Rx])
        try: d = np.linalg.solve(J, f)
        except np.linalg.LinAlgError: return None
        if not np.all(np.isfinite(d)): return None
        step = np.max(np.abs(d))
        if step > 3: d = d*3/step
        tau -= d[0]; x -= d[1]
        if abs(tau) > 1e3 or abs(x.imag) > 60 or not np.isfinite(tau): return None
        if np.max(np.abs(d)) < 1e-13: return tau, x
    return None

def refine(a, tau, x, dps=50):
    mp.mp.dps = dps
    f = lambda T, X: (Rfun(a, T, X), -sum((n+1)*an*mp.e**(-(n+1)**2*T/2)*mp.sin((n+1)*X) for n, an in enumerate(a)))
    s = mp.findroot(f, (mp.mpc(tau), mp.mpc(x)), tol=1e-30, maxsteps=50)
    return s[0], s[1]

def local_data(a, tau, x):
    """alpha=R_tau, beta=R_xx at the double zero; h_sing=(beta/3)(2 alpha (tau-tau*)/beta)^{3/2}."""
    al = -sum((n+1)**2/mp.mpf(2)*an*mp.e**(-(n+1)**2*tau/2)*mp.cos((n+1)*x) for n, an in enumerate(a))
    be = -sum((n+1)**2*an*mp.e**(-(n+1)**2*tau/2)*mp.cos((n+1)*x) for n, an in enumerate(a))
    return al, be

def canon_x(x):
    x = complex(x); re = x.real % (2*math.pi)
    if re > math.pi: return complex(2*math.pi-re, -x.imag)
    return complex(re, x.imag)
