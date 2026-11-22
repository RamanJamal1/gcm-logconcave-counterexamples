import time, math, torch
from wp import *
dev = 'cuda'
t = lambda v: torch.tensor([v], device=dev, dtype=DT)
# (a) Gaussian
for v in (1.0, 2.5):
    x = make_grid(1, t(14 * math.sqrt(v)).squeeze(0).expand(1), 4001)
    out, u2, r, d = evaluate(x, 'none', {'eps': t(1 / v)}, [], ms=(2, 3, 5, 8, 15, 16))
    for m, (S, A) in out.items():
        print('gauss v', v, 'm', m, float(S), math.factorial(m - 1) / (2 * v ** m), float(S) / A.item())
# (b) witness
def wit(npts, half):
    pk = [{'c': t(0.), 'w': t(1e6), 'omega': t(1.0), 'a': torch.tensor([[1 / 15, 1 / 40000, -1 / 18e6]], device=dev, dtype=DT),
           'theta': torch.zeros(1, 3, device=dev, dtype=DT)}]
    x = make_grid(1, t(half).expand(1), npts)
    # constant plateau: phi=exp(-(x)^2/(2e12)) ~ 1
    out, u2, r, d = evaluate(x, 'none', {'eps': t(1 / 14)}, pk, ms=(14, 15, 16))
    return out, u2, r
for half, npts in ((12 * math.sqrt(14), 20001), (14 * math.sqrt(14), 40001), (14 * math.sqrt(14), 80001)):
    out, u2, r = wit(npts, half)
    print(half, npts, {m: (float(S), float(S / A)) for m, (S, A) in out.items()}, float(u2.min()), float(r.min()))
# baseline 2: Gaussian x Fourier kappa
a = [0.0558, 1.86e-5, -3.99e-8, 2.65e-10, -2.85e-12]
kap = 0.0590
ar = torch.tensor([a], device=dev, dtype=DT)
# this family is exp(-kap x^2/2 + sum a_k cos kx) (potential family) -> not log(1+g); use g=exp(psi)-1 not needed: do via direct jets
def pot(npts=40001):
    x = make_grid(1, t(12 / math.sqrt(kap)).expand(1), npts)
    X = jvar(x)
    lp = -0.5 * kap * jmul(X, X)
    for k, ak in enumerate(a):
        lp = lp + ak * jcos_t(k + 1, 1.0, 0.0, x)
    u = {j: lp[..., j] * FACT[j] for j in range(1, N + 1)}
    dens = torch.exp(lp[..., 0] - lp[..., 0].max()); dens = dens / torch.trapezoid(dens, x, dim=-1)[..., None]
    for m in (14, 15, 16):
        F = INTEGRANDS[m](u)
        S = (-1) ** m * torch.trapezoid(F * dens, x, dim=-1); A = torch.trapezoid(F.abs() * dens, x, dim=-1)
        print('baseline2 m', m, float(S), float(S / A))
    print('min u2', float(u[2].min()))
pot()
# throughput
B = 256
x = make_grid(B, torch.full((B,), 12 * math.sqrt(14), device=dev, dtype=DT), 2049)
bp = {'eps': torch.full((B,), 1 / 14, device=dev, dtype=DT), 'A': torch.full((B,), 0.3, device=dev, dtype=DT), 'L': torch.full((B,), 3., device=dev, dtype=DT)}
pk = [{'c': torch.zeros(B, device=dev, dtype=DT), 'w': torch.full((B,), 8., device=dev, dtype=DT), 'omega': torch.ones(B, device=dev, dtype=DT),
       'a': torch.full((B, 3), 0.01, device=dev, dtype=DT), 'theta': torch.zeros(B, 3, device=dev, dtype=DT)}]
for _ in range(2):
    torch.cuda.synchronize(); t0 = time.time()
    out, *_ = evaluate(x, 'sqrt', bp, pk)
    torch.cuda.synchronize(); print('eval B=256 X=2049', time.time() - t0)
