"""Read a numerical SDP restart file without upgrading it to a proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np,scipy.linalg as la
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def inspect(m,source,output,sharp=False):
 tag="sharp_" if sharp else ""
 data=np.load(ROOT/f'checkpoints/native_sos_{tag}m{m}.npz');dims=list(map(int,data['dims']));scales=[data['scales0'].tolist(),data['scales1'].tolist()];qs=[np.zeros((n,n)) for n in dims];grams=[[['0']*n for _ in range(n)] for n in dims]
 for line in Path(source).read_text().splitlines()[1:]:
  k,block,i,j,value=line.split();k,block,i,j=map(int,[k,block,i,j])
  if k!=2:continue
  block-=1;i-=1;j-=1;qs[block][i,j]=qs[block][j,i]=float(value);v=(Fraction(value)+(Fraction(1,100000) if i==j else 0))*Fraction(math.factorial(m-1),scales[block][i]*scales[block][j]);grams[block][i][j]=grams[block][j][i]=str(v)
 x=np.concatenate([q.ravel() for q in qs]);residual=float(np.max(abs(data['A']@x-data['b'])));eigs=[float(la.eigvalsh(q+1e-5*np.eye(n))[0]) for q,n in zip(qs,dims)];r={'order':m,'proved':False,'status':'numerical_seed_only','normalization':math.factorial(m-1),'basis_scales':scales,'gram_matrices':grams,'min_scaled_eigenvalues':eigs,'residual':residual,'source':str(source)};write_json(output,r);print({k:v for k,v in r.items() if k not in ['basis_scales','gram_matrices']},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);p.add_argument('source');p.add_argument('output');p.add_argument('--sharp',action='store_true');a=p.parse_args();inspect(a.order,a.source,a.output,a.sharp)
