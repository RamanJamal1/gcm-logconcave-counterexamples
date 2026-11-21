import sys, json, time
from ts import *
name=sys.argv[1]; N=int(sys.argv[2]); grid=int(sys.argv[3]); dps=int(sys.argv[4])
t0=time.time(); cfg=CONFIGS[name]
h=h_coeffs(cfg['a'],N,grid,dps)
json.dump({'name':name,'N':N,'grid':grid,'dps':dps,'h':[mp.nstr(v,45) for v in h]},open('data/hn_%s_N%d_g%d.json'%(name,N,grid),'w'))
print(name,'done',time.time()-t0)
