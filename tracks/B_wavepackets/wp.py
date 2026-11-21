"""Track B: localized wave packets on convex bases. Taylor-jet evaluator (torch, FP64).

log p = -V(x) + log(1 + g(x)),  g = sum_packets phi_j(x) sum_k a_jk cos(k w_j x + th_jk)
All derivatives u_j = d^j log p computed by exact Taylor-jet arithmetic (differentiable in torch).
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
from generated_extended_integrands import INTEGRANDS
N = 16
DT = torch.float64
FACT = [math.factorial(k) for k in range(40)]


def jmul(a, b, n=N):
    return torch.stack([sum(a[..., i] * b[..., k - i] for i in range(k + 1)) for k in range(n + 1)], -1)


def jexp(a, n=N):
    e = [torch.exp(a[..., 0])]
    for k in range(1, n + 1):
        e.append(sum(j * a[..., j] * e[k - j] for j in range(1, k + 1)) / k)
    return torch.stack(e, -1)


def jlog(a, n=N):
    l = [torch.log(a[..., 0])]
    for k in range(1, n + 1):
        s = a[..., k]
        if k > 1:
            s = s - sum(j * l[j] * a[..., k - j] for j in range(1, k)) / k
        l.append(s / a[..., 0])
    return torch.stack(l, -1)


def jsqrt(a, n=N):
    s = [torch.sqrt(a[..., 0])]
    for k in range(1, n + 1):
        t = a[..., k]
        if k > 1:
            t = t - sum(s[j] * s[k - j] for j in range(1, k))
        s.append(t / (2 * s[0]))
    return torch.stack(s, -1)


def jvar(x, n=N):
    z = torch.zeros_like(x)
    return torch.stack([x, torch.ones_like(x)] + [z] * (n - 1), -1)


def jcos_t(k, w, th, x, n=N):
    ph = k * w * x + th
    c, s = torch.cos(ph), torch.sin(ph)
    cy = [c, -s, -c, s]
    return torch.stack([(k * w) ** j / FACT[j] * cy[j % 4] for j in range(n + 1)], -1)


def base_jet(x, base, bp, n=N):
    """-V jet.  bp: dict of tensors broadcastable to x (eps, A, L, gam, tilt)."""
    X = jvar(x, n)
    X2 = jmul(X, X, n)
    res = -0.5 * bp['eps'][..., None] * X2
    if 'gam' in bp:
        res = res - bp['gam'][..., None] * jmul(X2, X2, n)
    if 'tilt' in bp:
        res = res + bp['tilt'][..., None] * X
    if base == 'sqrt':
        L = bp['L']
        X2c = X2.clone()
        X2c[..., 0] = X2c[..., 0] + L * L
        res = res - bp['A'][..., None] * jsqrt(X2c, n)
    elif base == 'logcosh':
        Y = X / bp['L'][..., None]
        h = 0.5 * (jexp(Y, n) + jexp(-Y, n))
        res = res - bp['A'][..., None] * jlog(h, n)
    elif base == 'none':
        pass
    else:
        raise ValueError(base)
    return res


def packet_g(x, packets, n=N):
    g = None
    for P in packets:
        d = jvar(x, n)
        d[..., 0] = d[..., 0] - P['c']
        q = -0.5 * jmul(d, d, n) / (P['w'] ** 2)[..., None]
        if P.get('pow', 1) != 1:  # super-Gaussian plateau: exp(-(d^2/2w^2)^pow) via exp(-(.)^p) not jet-safe; unsupported
            raise ValueError
        phi = jexp(q, n)
        K = P['a'].shape[-1]
        tot = 0
        for k in range(K):
            tot = tot + P['a'][..., k, None] * jcos_t(k + 1, P['omega'], P['theta'][..., k], x, n)
        gp = jmul(phi, tot, n)
        g = gp if g is None else g + gp
    return g


def logp_jet(x, base, bp, packets, n=N):
    lp = base_jet(x, base, bp, n)
    if packets:
        g1 = packet_g(x, packets, n)
        g1 = g1.clone()
        g1[..., 0] = g1[..., 0] + 1
        lp = lp + jlog(g1, n)
        return lp, g1[..., 0]
    return lp, torch.ones_like(x)


def evaluate(x, base, bp, packets, ms=(14, 15, 16), n=N):
    """x [B,X] grid. Returns {m:(S_m, int|F_m|p)}, u2, r, density."""
    lp, r0 = logp_jet(x, base, bp, packets, n)
    u = {j: lp[..., j] * FACT[j] for j in range(1, n + 1)}
    lpv = lp[..., 0]
    dens = torch.exp(lpv - lpv.max(-1, keepdim=True).values)
    dens = dens / torch.trapezoid(dens, x, dim=-1)[..., None]
    out = {}
    for m in ms:
        F = INTEGRANDS[m](u)
        G = (-1) ** m * F
        out[m] = (torch.trapezoid(G * dens, x, dim=-1), torch.trapezoid(F.abs() * dens, x, dim=-1), torch.trapezoid(G.clamp_min(0) * dens, x, dim=-1))
    return out, u[2], r0, dens


def make_grid(B, half, npts, dev='cuda'):
    """half: tensor [B] half-width. uniform grid per row."""
    t = torch.linspace(-1, 1, npts, device=dev, dtype=DT)
    return half[:, None] * t[None, :]
