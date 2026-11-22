"""Build exact-rational constraint data in double-double (hi,lo) COO form.
A_dd(r,:) = A_exact(r,:)/(s_i s_j)/rn_r ; b_dd = (target/norm - 1e-5*A_scaled(r,:).vec(I))/rn_r  (exact Fractions -> DD)
Usage: build_dd.py m"""
import os,sys,time,math,pickle
from fractions import Fraction
import numpy as np
ROOT=os.path.expanduser("~/gcm-logconcave-counterexamples");m=int(sys.argv[1]);D=ROOT+"/tracks/D_order14/data";O=ROOT+"/tracks/E_ddipm/data"
def dd(f):
    h=float(f);return h,float(f-Fraction(h))
t0=time.time();P=pickle.load(open(f"{D}/setup_m{m}.pkl","rb"));S=np.load(f"{D}/sdp_m{m}.npz")
bases,target,keys,cols,layout=P["bases"],P["target"],P["keys"],P["cols"],P["layout"]
dims=[len(b) for b in bases];norm=math.factorial(m-1);sel=[int(r) for r in S["rows"]];selset={r:i for i,r in enumerate(sel)}
scales=[S["scales0"].tolist(),S["scales1"].tolist()];offs=[0,dims[0]**2];ix={k:i for i,k in enumerate(keys)}
ent={r:[] for r in sel}
for col,(blk,i,j) in zip(cols,layout):
    n=dims[blk];o=offs[blk];den=scales[blk][i]*scales[blk][j]
    for key,v in col.items():
        r=ix[key]
        if r not in selset: continue
        f=Fraction(int(v),den)
        if i==j: ent[r].append((o+i*n+j,f))
        else:
            ent[r].append((o+i*n+j,f/2));ent[r].append((o+j*n+i,f/2))
diag=set()
for blk in range(2):
    n=dims[blk];diag|={offs[blk]+i*n+i for i in range(n)}
R=[];C=[];H=[];L=[];bh=[];bl=[]
for k,r in enumerate(sel):
    e=ent[r];rn=max(math.sqrt(sum(float(f)**2 for _,f in e)),1e-300);frn=Fraction(rn)
    aI=sum((f for c,f in e if c in diag),Fraction(0))
    bf=(Fraction(str(target.get(keys[r],0)))/norm-Fraction(1,100000)*aI)/frn
    h,l=dd(bf);bh.append(h);bl.append(l)
    for c,f in e:
        h,l=dd(f/frn);R.append(k);C.append(c);H.append(h);L.append(l)
np.savez(f"{O}/dd_m{m}.npz",R=np.array(R,dtype=np.int32),C=np.array(C,dtype=np.int32),H=np.array(H),L=np.array(L),bh=np.array(bh),bl=np.array(bl),dims=np.array(dims),norm=norm,scales0=np.array(scales[0]),scales1=np.array(scales[1]))
print("built",m,len(sel),"rows nnz",len(R),"time",round(time.time()-t0,1),flush=True)
