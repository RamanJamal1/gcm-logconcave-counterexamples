import math,torch,numpy as np
from search import *
fam=Fam('sqrt',True,1,3)
x=torch.tensor(witness_x(fam)[None],device=dev,dtype=DT)
x[0,4]=math.log(1e5)
for npts in (6001,20001):
    out,margin,rmin,bp,pk=run_eval(fam,x,npts)
    print(npts,{m:float(out[m][0]/out[m][1]) for m in out if m!='var'},float(margin),float(rmin))
