import sys,json,math,numpy as np,torch
sys.path.insert(0,'.')
import kap
from wp import DT
res={}
for tag in sys.argv[1:]:
    d=json.load(open(f'data/cma_rho_{tag}.json')); x=np.array(d['x']); NF=d['NF']; pk=[]
    for j in range(NF):
        om=1.0 if j==0 else math.exp(x[3*NF+j-1]); y,z2,z3=x[3*j:3*j+3]
        a=[0.07*math.exp(y)/om**2,z2/(2*om)**15,z3/(3*om)**15]
        pk.append({'c':0.0,'omega':om,'a':a,'theta':[0,0,0]})
    cfg={'packets':pk}; json.dump(cfg,open(f'data/cfg_cma_rho_{tag}.json','w'))
    res[tag]={'cfg':cfg,'rho':kap.rho(tag,kap.packs_from(cfg))}
json.dump(res,open('data/cma_rho_refined.json','w'),indent=1)
