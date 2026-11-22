"""Principal-sheet test: fit exact g_n (n in [n0,400], 100 digits) with
   g_n = sum_j sigma_j^{-n} [A_j c32(n) + B_j c52(n)],  c_s(n)=Gamma(n-s)/(Gamma(-s) n!)  (coeff of (1-z)^s)
over ALL double-zero candidates (R=R_x=0) with |sigma|<=rho*|sigma_min| (complex x allowed).
A_j fitted vs A_j^pred = mult*(beta/3)*sqrt(q)^3 (branch eps=+-1).  |A_fit| ~ |A_pred| => principal sheet; ~0 => other sheet.
Numerical evidence only."""
import json, sys
from ts import *
from an2 import loadg
mp.mp.dps = 100

def cs(s, n): return mp.gamma(n-s)/(mp.gamma(-s)*mp.factorial(n))

def run(name, rho=1.6, n0=120, N=400):
    g = loadg(name, N, 128)
    cfg = CONFIGS[name]; a = cfg['a']; kap = cfg['kappa']
    d = json.load(open('data/sing_%s.json' % name))
    cand = []
    for o in d:
        tau = mp.mpc(*o['tau']); x = mp.mpc(*o['x']); sig = -tau
        cand.append((abs(sig), sig, x))
    cand.sort(key=lambda c: c[0]); smin = cand[0][0]
    cand = [c for c in cand if c[0] <= rho]
    # refine in 100 digits
    rows = []; seen = {}
    for _, sig, x in cand:
        T, X = refine(a, -sig, x, dps=100)
        key = (round(float((-T).real), 6), round(float((-T).imag), 6))
        if key in seen: seen[key][2].append(X); continue
        seen[key] = [-T, X, [X]]
    rows = [(v[0], v[2]) for v in seen.values()]
    ncol = 2*len(rows); ns = list(range(n0, N+1))
    A = mp.matrix(len(ns), ncol); b = mp.matrix(len(ns), 1)
    for i, n in enumerate(ns):
        b[i] = g[n]
        for j, (sig, x) in enumerate(rows):
            A[i, 2*j] = sig**(-n)*cs(mp.mpf(3)/2, n); A[i, 2*j+1] = sig**(-n)*cs(mp.mpf(5)/2, n)
    # column scaling for conditioning
    sc = [mp.sqrt(sum(abs(A[i, j])**2 for i in range(len(ns)))) for j in range(ncol)]
    for j in range(ncol):
        for i in range(len(ns)): A[i, j] /= sc[j]
    sol, res = mp.qr_solve(A, b)
    out = []
    for j, (sig, x) in enumerate(rows):
        Aj = sol[2*j]/sc[2*j]
        Ap = 0; mult = 0
        for xx in x:
            al, be = local_data(a, -sig, xx); q = 2*al*sig/be
            re = float(canon_x(xx).real); mu = 1 if (abs(re) < 1e-8 or abs(re-math.pi) < 1e-8) else 2
            Ap += mu*(be/3)*mp.sqrt(q)**3; mult += mu
        x = x[0]
        ratio = Aj/Ap
        Tt = sig/(1+kap*sig)
        out.append(dict(sigma=[float(sig.real), float(sig.imag)], x=[float(x.real), float(x.imag)], T=[float(Tt.real), float(Tt.imag)],
                        A_fit=[float(Aj.real), float(Aj.imag)], A_pred=[float(Ap.real), float(Ap.imag)], ratio=[float(ratio.real), float(ratio.imag)], mult=mult))
    print(name, 'residual', mp.nstr(res, 3), 'ncand', len(rows))
    for o in out:
        print('sigma=%+.4f%+.4fi x=%+.3f%+.3fi |sig|=%.4f  A_fit/A_pred=%+.3f%+.3fi  |A_pred|=%.3g' % (o['sigma'][0], o['sigma'][1], o['x'][0], o['x'][1], abs(complex(*o['sigma'])), o['ratio'][0], o['ratio'][1], abs(complex(*o['A_pred']))))
    json.dump(out, open('data/fit_%s.json' % name, 'w'), indent=1)

if __name__ == '__main__':
    run(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.6, int(sys.argv[3]) if len(sys.argv) > 3 else 120)
