import sys, json, numpy as np, random, math
import mpmath as mp
from ts import *

def search(name, nseed=6000, seed=1):
    cfg = CONFIGS[name]; a = cfg['a']; rng = random.Random(seed)
    found = {}
    for i in range(nseed):
        tau0 = complex(rng.uniform(-14, 4), rng.uniform(-40, 40)); x0 = complex(rng.uniform(0, math.pi), rng.uniform(-14, 14))
        r = newton_sing(a, tau0, x0)
        if r is None: continue
        tau, x = r
        if abs(tau) > 80: continue
        xc = canon_x(x)
        key = (round(tau.real, 5), round(tau.imag, 5), round(xc.real, 4), round(xc.imag, 4))
        found[key] = (tau, xc)
    out = []
    for key, (tau, x) in found.items():
        try:
            tt, xx = refine(a, tau, x)
        except Exception:
            continue
        al, be = local_data(a, tt, xx)
        t = tt/(1-cfg['kappa']*tt)
        out.append(dict(tau=complex(tt), x=complex(xx), t=complex(t), abs_tau=float(abs(tt)), abs_t=float(abs(t)), alpha=complex(al), beta=complex(be)))
    ded = {}
    for o in out:
        xc = canon_x(o['x']); k = (round(o['tau'].real, 7), round(o['tau'].imag, 7), round(xc.real, 5), round(xc.imag, 5))
        o['x'] = xc; ded[k] = o
    return sorted(ded.values(), key=lambda o: o['abs_t'])

def dump(name, res):
    ser = [{k: ([v.real, v.imag] if isinstance(v, complex) else v) for k, v in o.items()} for o in res]
    json.dump(ser, open('data/sing_%s.json' % name, 'w'), indent=1)

if __name__ == '__main__':
    name = sys.argv[1]; res = search(name, int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
    dump(name, res)
    print(name, len(res))
    for o in res[:16]:
        print('tau=%+.5f%+.5fi  x=%+.4f%+.4fi  t=%+.5f%+.5fi |t|=%.4f |tau|=%.4f' % (o['tau'].real, o['tau'].imag, o['x'].real, o['x'].imag, o['t'].real, o['t'].imag, o['abs_t'], o['abs_tau']))
