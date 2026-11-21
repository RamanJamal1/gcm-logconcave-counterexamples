"""Order-m sparse SOS feasibility data (same construction as src/prepare_sparse_sdp.py),
written to tracks/D_order14/data/ so no existing checkpoint is touched; plus a numerically
chosen set of linearly independent rows (the exact verifier later re-checks everything)."""
import os,sys,time,math,json,pickle
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,scipy.sparse as sp,scipy.linalg as la
ROOT=os.path.expanduser('~/gcm-logconcave-counterexamples');sys.path.insert(0,ROOT+'/src')
from sos_quotient import setup
m=int(sys.argv[1]);out=ROOT+'/tracks/D_order14/data';os.makedirs(out,exist_ok=True)
t=time.time();bases,target,keys,cols,layout,rows=setup(m);print('setup',round(time.time()-t,1),'s',flush=True)
pickle.dump({'bases':bases,'target':target,'keys':keys,'cols':cols,'layout':layout},open(f'{out}/setup_m{m}.pkl','wb'))
dims=list(map(len,bases));norm=math.factorial(m-1)
scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
offsets=[0,dims[0]**2];ix={k:i for i,k in enumerate(keys)};R,C,V=[],[],[]
for col,(block,i,j) in zip(cols,layout):
    n=dims[block];o=offsets[block]
    for key,v in col.items():
        v=float(v)/(scales[block][i]*scales[block][j]);r=ix[key]
        if i==j:R.append(r);C.append(o+i*n+j);V.append(v)
        else:R+= [r,r];C+=[o+i*n+j,o+j*n+i];V+=[v/2,v/2]
N=sum(n*n for n in dims);A=sp.csr_matrix((V,(R,C)),shape=(len(keys),N));A.sum_duplicates()
b=np.array([float(target.get(k,0))/norm for k in keys])
rn=np.maximum(np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel()),1e-300);D=sp.diags(1/rn);A=(D@A).tocsr();b=b/rn
print('rows',A.shape[0],'cols',N,'nnz',A.nnz,'dims',dims,flush=True)
t=time.time();Q,Rr,piv=la.qr(A.toarray().T,mode='economic',pivoting=True);d=np.abs(np.diag(Rr))
rank=int((d>d[0]*1e-11).sum());sel=np.sort(piv[:rank]);print('numeric rank',rank,'qr',round(time.time()-t,1),'s; diag ratio at rank',d[rank-1]/d[0],flush=True)
identity=np.concatenate([np.eye(n).ravel() for n in dims])
As=A[sel];bs=b[sel]-1e-5*(As@identity)
sp.save_npz(f'{out}/A_m{m}.npz',As);np.savez(f'{out}/sdp_m{m}.npz',b=bs,c=identity,rows=sel,dims=dims,scales0=scales[0],scales1=scales[1],norm=norm)
full_resid_rows=A.shape[0]-rank
json.dump({'order':m,'rows_total':A.shape[0],'rank_numeric':rank,'cols':N,'nnz_selected':int(As.nnz),'dims':dims,'margin':1e-5,'proved':False,'note':'numeric row selection; exact verifier re-checks the full identity'},open(f'{out}/prep_m{m}.json','w'),indent=2)
print('done',flush=True)
