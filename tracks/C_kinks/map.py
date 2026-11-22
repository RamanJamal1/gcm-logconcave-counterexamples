import sys, json; sys.path.insert(0,'/tmp')
from kinks import *
ds=np.round(np.geomspace(0.1,20,40),4)
fam={
 'k2_sym_a0.3':lambda d:([-d/2,d/2],[-.3,0,.3]),
 'k2_sym_a1':lambda d:([-d/2,d/2],[-1,0,1]),
 'k2_sym_a3':lambda d:([-d/2,d/2],[-3,0,3]),
 'k2_asym_-1_0_3':lambda d:([-d/2,d/2],[-1,0,3]),
 'k2_steepmid_-1_-0.2_1':lambda d:([-d/2,d/2],[-1,-.2,1]),
 'k3_eq_a1':lambda d:([-d,0,d],[-1,-1/3,1/3,1]),
 'k3_eq_a3':lambda d:([-d,0,d],[-3,-1,1,3]),
 'k3_asym':lambda d:([-d,0,2*d],[-2,-.5,.4,1.5]),
}
out={}
for k,fn in fam.items():
    rows=[]
    for d in ds:
        xs,ss=fn(float(d))
        r=S_R(xs,ss,ms=range(6,17),h=0.05,margin=13)
        rows.append({'d':float(d),'R':{m:float(r[m][2]) for m in r}})
    out[k]=rows
    mn=min((row['R'][m],row['d'],m) for row in rows for m in row['R'] if m>=14)
    print(k,'min R over d for m>=14:',mn,flush=True)
json.dump(out,open('/tmp/map_k2k3.json','w'))
