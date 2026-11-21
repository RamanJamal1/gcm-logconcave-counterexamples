import math,torch,numpy as np
from search import *
fam=Fam('sqrt',True,1,3)
x=torch.tensor(witness_x(fam)[None],device=dev,dtype=DT)
x[0,4]=math.log(1e5)
bp,pk=fam.decode(x)
print({k:v.flatten().tolist() for k,v in bp.items()}, {k:v.flatten().tolist() for k,v in pk[0].items()})
xx=make_grid(1,torch.tensor([48.6],device=dev,dtype=DT),20001)
for base in ('sqrt','none'):
  out,u2,r,d=evaluate(xx,base,bp,pk)
  print(base,{m:(float(out[m][0]),float(out[m][1])) for m in out})
