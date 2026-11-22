"""High-precision (Arb series, 200 bit) evaluation of S_m for an explicit config via exact sparse integrand + trapezoid.
usage: python val_mp.py cfg.json npts [half]   (cfg: base, eps, A, L, packets[{c,w,omega,a[],theta[]}])"""
import sys, json, math, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from fractions import Fraction
from flint import arb, arb_series, ctx
from entropy_sparse import signed_polynomial
PREC = 200
ctx.prec = PREC
ctx.cap = 18
NJ = 16


def A(v): return arb(v)


def lp_series(x, cfg):
    X = arb_series([x, 1] + [0] * (NJ - 1), prec=PREC)
    eps = A(cfg['eps'])
    res = -(eps / 2) * X * X
    if cfg.get('gam', 0):
        res = res - A(cfg['gam']) * X * X * X * X
    if cfg.get('tilt', 0):
        res = res + A(cfg['tilt']) * X
    if cfg['base'] == 'sqrt' and cfg['A'] > 0:
        L = A(cfg['L'])
        res = res - A(cfg['A']) * (X * X + L * L).sqrt()
    elif cfg['base'] == 'logcosh' and cfg['A'] > 0:
        Y = X / A(cfg['L'])
        res = res - A(cfg['A']) * ((Y.exp() + (-Y).exp()) / 2).log()
    g = arb_series([0] * (NJ + 1), prec=PREC)
    for P in cfg['packets']:
        d = X - A(P['c'])
        phi = (-(d * d) / (2 * A(P['w']) ** 2)).exp()
        s = arb_series([0] * (NJ + 1), prec=PREC)
        for k, (ak, th) in enumerate(zip(P['a'], P['theta']), 1):
            ph = (k * A(P['omega'])) * X + A(th)
            s = s + A(ak) * ph.cos()
        g = g + phi * s
    res = res + (g + 1).log()
    return res


def evaluate(cfg, m, npts, half):
    terms = [(e, arb(c.numerator) / c.denominator) for e, c in signed_polynomial(m).items()]
    # ensure sign convention: signed_polynomial includes (-1)^m
    xs = [A(-half) + (2 * A(half)) * i / (npts - 1) for i in range(npts)]
    dx = 2 * A(half) / (npts - 1)
    lps, nums = [], []
    facts = [math.factorial(j) for j in range(NJ + 1)]
    for x in xs:
        s = lp_series(x, cfg)
        co = [s[j] for j in range(NJ + 1)]
        u = {j + 1: co[j + 1] * facts[j + 1] for j in range(NJ)}
        upow = {}
        val = arb(0)
        for e, c in terms:
            t = c
            for i, pw in enumerate(e):
                if pw:
                    key = (i + 1, pw)
                    if key not in upow:
                        upow[key] = u[i + 1] ** pw
                    t = t * upow[key]
            val += t
        lps.append(co[0]); nums.append(val)
    mx = max(l.mid() for l in lps)
    ws = [(l - mx).exp() for l in lps]
    def trap(vals):
        return dx * (sum(vals[1:-1], arb(0)) + (vals[0] + vals[-1]) / 2)
    Z = trap(ws)
    return trap([w * n for w, n in zip(ws, nums)]) / Z


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1])); npts = int(sys.argv[2]); half = float(sys.argv[3])
    ms = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [15]
    for m in ms:
        t0 = time.time()
        print(m, evaluate(cfg, m, npts, half).str(25), time.time() - t0, flush=True)
