import math,torch,numpy as np
from wp import *
dev='cuda'
# gaussian eps=1 (v=1), single harmonic cos(om x), amplitude a ; scan om,a. report S_m/S_m^G - 1 for m=14..17
oms=[0.5,1,1.5,2,2.5,3,3.5,4,5,6]
for m in (14,15,16,17):
    SG=math.factorial(m-1)/2
    print('m',m)
    for om in oms:
        row=[]
        for a in (1e-4,1e-3,1e-2):
            B=1
            x=make_grid(1,torch.tensor([14.],device=dev,dtype=DT),40001)
            pk=[{'c':torch.zeros(1,1,device=dev,dtype=DT),'w':torch.full((1,1),1e4,device=dev,dtype=DT),'omega':torch.full((1,1),om,device=dev,dtype=DT),'a':torch.full((1,1,1),a,device=dev,dtype=DT),'theta':torch.zeros(1,1,1,device=dev,dtype=DT)}]
            out,*_=evaluate(x,'none',{'eps':torch.ones(1,1,device=dev,dtype=DT)},pk,ms=(m,))
            row.append(float((out[m][0]/SG-1)/a**2))
        print(om,['%.3e'%v for v in row])
