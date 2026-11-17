"""Prepare an unwhitened sparse-coefficient SDP for a higher-order sign proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,math,json
import numpy as np
from sos_quotient import setup,ROOT

def prepare(m):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);dims=list(map(len,bases));norm=math.factorial(m-1);scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases];offsets=[0,dims[0]**2];A=np.zeros((len(keys),sum(n*n for n in dims)));b=np.array([float(target.get(k,0))/norm for k in keys]);ix={k:i for i,k in enumerate(keys)}
 for col,(block,i,j) in zip(cols,layout):
  n=dims[block];offset=offsets[block]
  for key,v in col.items():
   v=float(v)/(scales[block][i]*scales[block][j]);r=ix[key]
   if i==j:A[r,offset+i*n+j]+=v
   else:A[r,offset+i*n+j]+=v/2;A[r,offset+j*n+i]+=v/2
 rn=np.maximum(np.linalg.norm(A,axis=1),1e-100);A/=rn[:,None];b/=rn;identity=np.concatenate([np.eye(n).ravel() for n in dims]);b-=1e-5*(A@identity);c=identity
 path=ROOT/f'checkpoints/native_sos_m{m}.npz';assert not path.exists(),'Preserve any existing solver matrix';np.savez(path,A=A,b=b,c=c,scales0=scales[0],scales1=scales[1],dims=dims)
 r={'order':m,'dimensions':dims,'rows':len(keys),'nonzero_entries':int(np.count_nonzero(A)),'proved':False,'method':'Original sparse exact coefficient constraints, row-scaled in float64; no dense whitening','seconds':time.time()-start};(ROOT/f'certs/sparse_sdp_preparation_m{m}.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);a=p.parse_args();prepare(a.order)
