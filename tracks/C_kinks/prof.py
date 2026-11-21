import sys,json,glob; sys.path.insert(0,'/tmp')
from ext import *
out={}
for f in sorted(glob.glob('/tmp/opt_*_15_1.json')):
    d=json.load(open(f)); r=S_R_ext(d['xs'],d['ss'],ms=range(6,21),h=0.05,margin=13)
    out[f]={'xs':d['xs'],'ss':d['ss'],'R':{m:r[m][2] for m in r}}
    print(f,' '.join('%d:%+.2e'%(m,r[m][2]) for m in r),flush=True)
json.dump(out,open('/tmp/profile_opt15.json','w'))
