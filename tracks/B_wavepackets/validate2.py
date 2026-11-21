import time, math, torch
from wp import *
dev = 'cuda'
t = lambda v: torch.tensor([v], device=dev, dtype=DT)
B = 256
c=lambda v: torch.full((B,1), v, device=dev, dtype=DT)
x = make_grid(B, torch.full((B,), 12 * math.sqrt(14), device=dev, dtype=DT), 2049)
bp = {'eps': c(1/14), 'A': c(.3), 'L': c(3.)}
pk = [{'c': c(0.), 'w': c(8.), 'omega': c(1.), 'a': torch.full((B,1, 3), 0.01, device=dev, dtype=DT), 'theta': torch.zeros(B,1, 3, device=dev, dtype=DT)}]
for _ in range(2):
    torch.cuda.synchronize(); t0 = time.time()
    out, *_ = evaluate(x, 'sqrt', bp, pk)
    torch.cuda.synchronize(); print('eval B=256 X=2049', time.time() - t0, {m:float(v[0][0]/v[1][0]) for m,v in out.items()})
