"""usage: finalize.py data/search_X.json  -> writes data/final_X.json with grid convergence + mp check"""
import sys, json, math, subprocess, time
import numpy as np, torch
from search import *

f = sys.argv[1]
d = json.load(open(f))
a = d['family']
fam = Fam(a['base'], bool(a['sym']), a['npk'], a['K'], bool(a['tilt']))
x = torch.tensor(np.array(d['x'])[None], device=dev, dtype=DT)
bp, packs = fam.decode(x)
cfg = {'base': fam.base if fam.base != 'quartic' else 'none', 'eps': float(bp['eps']), 'A': float(bp['A']), 'L': float(bp['L']),
       'gam': float(bp['gam']) if 'gam' in bp else 0.0, 'tilt': float(bp['tilt']) if 'tilt' in bp else 0.0,
       'packets': [{k: (v.flatten().tolist() if k in ('a', 'theta') else float(v.flatten()[0])) for k, v in P.items()} for P in packs]}
res = {'source': f, 'cfg': cfg, 'search': {k: d[k] for k in d if k not in ('x',)}, 'x': d['x'], 'grids': []}
half0 = 13 / math.sqrt(cfg['eps'])
for mult, npts in ((1, 2500), (1, 6001), (1, 20001), (1.3, 40001), (1.3, 80001)):
    half = torch.tensor([half0 * mult], device=dev, dtype=DT)
    xx = make_grid(1, half, npts)
    out, u2, r, dens = evaluate(xx, cfg['base'], {k: torch.tensor([[v]], device=dev, dtype=DT) for k, v in cfg.items() if k in ('eps', 'A', 'L', 'gam', 'tilt')}, packs, ms=(14, 15, 16))
    dx = float(xx[0, 1] - xx[0, 0]); mean_u2 = float((u2 * dens).sum() * dx)
    row = {'half': half0 * mult, 'npts': npts, 'S14': float(out[14][0]), 'S15': float(out[15][0]), 'S16': float(out[16][0]),
           'R14': float(out[14][0] / out[14][1]), 'R15': float(out[15][0] / out[15][1]), 'R16': float(out[16][0] / out[16][1]),
           'max_u2': float(u2.max()), 'margin': float(-u2.max() / (-mean_u2)), 'min_r': float(r.min())}
    res['grids'].append(row); print(row, flush=True)
name = f.split('search_')[1].replace('.json', '')
json.dump(cfg, open(f'data/cfg_{name}.json', 'w'))
t0 = time.time()
p = subprocess.run(['python3', 'val_mp.py', f'data/cfg_{name}.json', '2500', str(half0), '15,16'], capture_output=True, text=True, timeout=110)
res['mp_check_npts2500'] = p.stdout.strip().splitlines(); print(p.stdout, p.stderr[-300:])
json.dump(res, open(f'data/final_{name}.json', 'w'), indent=1)
