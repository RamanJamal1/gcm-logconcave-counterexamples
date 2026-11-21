import sys, json, time, math, argparse, os
import numpy as np, torch, cma
from wp import *

dev = 'cuda'
R0 = 1e-10
MARGIN = float(os.environ.get("MARGIN", 0.02))
SCALES = np.array([0.3, 0.03, 0.003, 3e-4, 3e-5])


class Fam:
    """name: base_sym_npk. params layout documented in decode()."""
    def __init__(self, base, sym, npk, K=3, tilt=False):
        self.base, self.sym, self.npk, self.K = base, sym, npk, K
        self.tilt = tilt and not sym
        self.names = ['logeps', 'logA', 'logL']
        for j in range(npk):
            self.names += [f'c{j}', f'logw{j}', f'logom{j}'] + [f'z{j}_{k}' for k in range(K)]
            if not sym:
                self.names += [f'th{j}_{k}' for k in range(K)]
        if self.tilt:
            self.names += ['tilt']
        self.D = len(self.names)

    def decode(self, X):
        """X: [B,D] torch -> (bp, packets)"""
        B = X.shape[0]
        i = 0
        col = lambda: X[:, None, :][:, :, 0:0]  # unused
        g = lambda j: X[:, j:j + 1]
        eps = torch.exp(g(0)).clamp(0.02, 3.0)
        A = torch.exp(g(1)).clamp(1e-7, 1e3)
        L = torch.exp(g(2)).clamp(1e-2, 1e3)
        bp = {'eps': eps, 'A': A, 'L': L}
        if self.base == 'quartic':
            bp['gam'] = A * 1e-2; bp['A'] = A * 0
        i = 3
        packs = []
        sc = torch.tensor(SCALES[:self.K], device=X.device, dtype=DT)
        for j in range(self.npk):
            c = g(i); w = torch.exp(g(i + 1)).clamp(0.2, 1e3)
            if os.environ.get('FIXW'): w = w * 0 + float(os.environ['FIXW'])
            om = torch.exp(g(i + 2)).clamp(0.05, 4.0)
            zr = X[:, i + 3:i + 3 + self.K]
            kk = torch.arange(1, self.K + 1, device=X.device, dtype=DT)[None, :]
            z = zr / (kk * om) ** 15
            z = torch.cat([torch.exp(zr[:, :1]) * eps / om ** 2, z[:, 1:]], 1)
            i += 3 + self.K
            if self.sym:
                th = torch.zeros_like(z)
            else:
                th = X[:, i:i + self.K]; i += self.K
            if self.sym and self.npk == 1:
                c = c * 0
            packs.append({'c': c, 'w': w, 'omega': om, 'a': z[:, None, :].expand(B, 1, self.K).reshape(B, 1, self.K) if False else z.unsqueeze(1), 'theta': th.unsqueeze(1)})
            if self.sym and self.npk >= 1 and j >= 0 and not (self.npk == 1):
                packs.append({'c': -c, 'w': w, 'omega': om, 'a': z.unsqueeze(1), 'theta': th.unsqueeze(1)})
        if self.tilt:
            bp['tilt'] = g(i)
        return bp, packs


def run_eval(fam, X, npts=8001, ms=(14, 15, 16)):
    bp, packs = fam.decode(X)
    half = 13.0 / bp['eps'][:, 0].sqrt()
    # for A large tails are thinner; keep generous
    x = make_grid(X.shape[0], half, npts, X.device)
    # shapes: parameters [B,1]; packet a [B,1,K] -> in packet_g a[...,k,None] -> [B,1,1]
    out, u2, r, dens = evaluate(x, fam.base if fam.base != 'quartic' else 'none', bp, packs, ms=ms)
    dxs = (x[:, 1] - x[:, 0])
    mean_u2 = (u2 * dens).sum(-1) * dxs
    mu = (x * dens).sum(-1) * dxs
    var = (((x - mu[:, None]) ** 2) * dens).sum(-1) * dxs
    out['var'] = var
    margin = -u2.max(-1).values / (-mean_u2).clamp_min(1e-30)
    return out, margin, r.min(-1).values, bp, packs


def fitness(fam, X, npts=8001):
    with torch.no_grad():
        out, margin, rmin, *_ = run_eval(fam, X, npts)
        S, A, _G = out[15]
        R = S / A
        Sh = S * out['var'] ** 15 / (math.factorial(14) / 2)
        rho = S / out[15][2]
        obj = torch.asinh(rho / R0)
        v = (MARGIN - margin).clamp_min(0) + (0.5 - rmin).clamp_min(0)
        bad = ~torch.isfinite(obj)
        obj = torch.where(bad, torch.full_like(obj, 1e6), obj)
        f = torch.where(v > 0, 100 + 1000 * v + obj.clamp_min(-50) * 0, obj)
        f = torch.where(bad, torch.full_like(f, 1e6), f)
        return f, R, margin, rmin, out


def witness_x(fam):
    x = np.zeros(fam.D)
    x[0] = math.log(1 / 14); x[1] = -14; x[2] = 0
    i = 3
    z = np.array([math.log(14 / 15), 2.5e-5 * 2 ** 15, -1 / 18e6 * 3 ** 15])
    x[i] = 0; x[i + 1] = math.log(300.); x[i + 2] = 0
    x[i + 3:i + 6] = z
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='sqrt'); ap.add_argument('--sym', type=int, default=1)
    ap.add_argument('--npk', type=int, default=1); ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--tilt', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--pop', type=int, default=96); ap.add_argument('--starts', type=int, default=6)
    ap.add_argument('--sigma', type=float, default=0.5); ap.add_argument('--npts', type=int, default=8001)
    ap.add_argument('--prior', type=int, default=3000)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    fam = Fam(a.base, bool(a.sym), a.npk, a.K, bool(a.tilt))
    rng = np.random.default_rng(a.seed)
    name = f"{os.environ.get('FIXW','loc')}m{MARGIN}_{a.base}_{'sym' if a.sym else 'asym'}_{a.npk}pk_K{a.K}{'_tilt' if a.tilt else ''}{a.tag}_s{a.seed}"
    t0 = time.time(); nev = 0
    # prior sampling
    def sample(n):
        X = np.zeros((n, fam.D))
        X[:, 0] = rng.uniform(math.log(0.02), math.log(1.0), n)
        X[:, 1] = rng.uniform(-6, 3, n)
        X[:, 2] = rng.uniform(math.log(0.3), math.log(30), n)
        i = 3
        for j in range(fam.npk):
            X[:, i] = rng.uniform(-10, 10, n) if (not fam.sym or fam.npk > 1) else 0
            X[:, i + 1] = rng.uniform(math.log(1), math.log(40), n)
            X[:, i + 2] = rng.uniform(math.log(0.3), math.log(3), n)
            X[:, i + 3:i + 3 + fam.K] = rng.normal(0, 1, (n, fam.K)); X[:, i + 3] = rng.normal(-0.4, 0.6, n)
            i += 3 + fam.K
            if not fam.sym:
                X[:, i:i + fam.K] = rng.uniform(-math.pi, math.pi, (n, fam.K)); i += fam.K
        if fam.tilt:
            X[:, i] = rng.normal(0, 0.05, n)
        return X
    P = sample(a.prior)
    fs = []
    for s in range(0, a.prior, 128):
        f, *_ = fitness(fam, torch.tensor(P[s:s + 128], device=dev, dtype=DT), a.npts); fs.append(f.cpu().numpy()); nev += min(128, a.prior - s)
    fs = np.concatenate(fs)
    print(name, 'prior feasible frac', float((fs < 90).mean()), 'best', float(fs.min()), flush=True)
    order = np.argsort(fs)
    starts = [P[i] for i in order[:a.starts]]
    if a.base == 'sqrt' and fam.sym and fam.npk == 1 and fam.K == 3:
        starts.append(witness_x(fam))
    best = {'f': 1e9}
    for si, x0 in enumerate(starts):
        es = cma.CMAEvolutionStrategy(x0, a.sigma, {'popsize': a.pop, 'seed': a.seed * 100 + si, 'verbose': -9})
        for it in range(a.iters):
            Xs = np.array(es.ask())
            f, R, margin, rmin, out = fitness(fam, torch.tensor(Xs, device=dev, dtype=DT), a.npts)
            nev += len(Xs)
            fn = f.cpu().numpy()
            es.tell(Xs, fn)
            j = int(fn.argmin())
            if fn[j] < best['f']:
                best = {'f': float(fn[j]), 'x': Xs[j].tolist(), 'R15': float(R[j]), 'rho15': float(out[15][0][j]/out[15][2][j]), 'rho14': float(out[14][0][j]/out[14][2][j]), 'R14': float(out[14][0][j] / out[14][1][j]), 'Shat15': float(out[15][0][j]*out['var'][j]**15/(math.factorial(14)/2)), 'S15': float(out[15][0][j]), 'S16': float(out[16][0][j]), 'var': float(out['var'][j]),
                        'margin': float(margin[j]), 'rmin': float(rmin[j]), 'start': si, 'it': it}
        print(name, 'start', si, 'best_this', es.result.fbest, 'global', best['f'], 'rho15', best['rho15'], 'Shat15', best['Shat15'], 'R15', best['R15'], 'R14', best['R14'], 'margin', best['margin'], f'{time.time()-t0:.0f}s', flush=True)
        best['names'] = fam.names; best['family'] = vars(a); best['nevals'] = nev; best['seconds'] = time.time() - t0
        json.dump(best, open(f'data/search_{name}.json', 'w'))
    print('DONE', name, best['R15'], best['R14'], nev, time.time() - t0, flush=True)

if __name__ == '__main__':
    main()
