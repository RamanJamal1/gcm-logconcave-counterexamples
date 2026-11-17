"""Exact rational row elimination before floating SDP conditioning.

This locator never reports a proof; reconstruction and independent verification
are mandatory even if the numerical solver reports positive definiteness.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,math
import sdpap
import numpy as np,scipy.linalg as la,scipy.sparse as sparse,sympy as s,cvxpy as cp
from sos_quotient import setup,ROOT

def probe(m):
 start=time.time();bases,target,keys,cols,layout,rows=setup(m);norm=math.factorial(m-1);dims=[len(b) for b in bases]
 scales=[[max(1,math.ceil(math.sqrt(math.prod(range(1,2*e[0],2))*math.prod(math.factorial(j)**n for j,n in enumerate(e) if j>0)))) for e in base] for base in bases]
 A=s.Matrix([[d.get(k,0) for d in cols] for k in keys]);b=s.Matrix([target.get(k,0) for k in keys]);factors=np.array([1/(scales[bl][i]*scales[bl][j]) for bl,i,j in layout]);num=np.array(A,float)*factors;rn=np.maximum(np.linalg.norm(num,axis=1),1e-100)
 single={};keyindex={k:i for i,k in enumerate(keys)}
 for j,d in enumerate(cols):
  if len(d)==1:
   row=keyindex[next(iter(d))];strength=abs(num[row,j])
   if row not in single or strength>single[row][0]:single[row]=(strength,j)
 chosen=[single[row][1] for row in sorted(single)];chosen_set=set(chosen);rest=[j for j in range(len(cols)) if j not in chosen_set];remaining=[i for i in range(len(keys)) if i not in single]
 _,_,tail=la.qr((num/rn[:,None])[remaining][:,rest],mode='economic',pivoting=True);permutation=chosen+[rest[int(j)] for j in tail]
 print('begin exact RREF',m,len(keys),len(cols),'singleton pivots',len(chosen),time.time()-start,flush=True)
 R,pivots=A[:,permutation].row_join(b).rref();assert len(cols) not in pivots;rank=len(pivots)
 print('exact RREF complete',rank,time.time()-start,flush=True)
 G=np.array(R[:rank,:-1],float)*factors[permutation];bb=np.array(R[:rank,-1],float).ravel()/norm;rn=np.maximum(np.linalg.norm(G,axis=1),1e-100);G/=rn[:,None];bb/=rn
 U,T=la.qr(G.T,mode='economic');bb=la.solve_triangular(T.T,bb,lower=True);G=U.T;condition=float(np.linalg.cond(T));print('condition after exact row elimination',condition,flush=True)
 count=sum(n*n for n in dims);offsets=[0,dims[0]**2];native=np.zeros((rank,count));shift=np.zeros(rank)
 for column,j in enumerate(permutation):
  bl,i,k=layout[j];offset=offsets[bl];n=dims[bl]
  if i==k:native[:,offset+i*n+k]=G[:,column];shift+=G[:,column]
  else:native[:,offset+i*n+k]=G[:,column]/2;native[:,offset+k*n+i]=G[:,column]/2
 delta=1e-5;rhs=bb-delta*shift;objective=np.zeros(count)
 for offset,n in zip(offsets,dims):
  for j in range(n):objective[offset+j*n+j]=1
 np.savez(ROOT/f'checkpoints/native_sos_m{m}.npz',A=native,b=rhs,c=objective,scales0=scales[0],scales1=scales[1],dims=dims)
 print('native multiprecision solve',time.time()-start,flush=True)
 x,y,info,timing,extra=sdpap.solve(sparse.csc_matrix(native),rhs,objective,sdpap.SymCone(s=tuple(dims)),sdpap.SymCone(f=rank),{'epsilonStar':1e-24,'epsilonDash':1e-24,'mpfPrecision':256,'maxIteration':200,'numThreads':1,'lambdaStar':1000.0,'upperBound':1e12,'lowerBound':-1e12,'print':'display'})
 xx=np.asarray(x.toarray()).ravel();grams=[];eigs=[]
 for offset,n,ss in zip(offsets,dims,scales):
  Q=xx[offset:offset+n*n].reshape(n,n)+delta*np.eye(n);grams.append((Q*norm/np.outer(ss,ss)).tolist());eigs.append(float(la.eigvalsh(Q)[0]))
 result={'order':m,'status':str(info['phasevalue']),'proved':False,'normalization':norm,'basis_scales':scales,'condition':condition,'seconds':time.time()-start,'gram_matrices':grams,'min_scaled_eigenvalues':eigs,'residual':float(np.max(abs(native@xx-rhs)))}
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=12);a=p.parse_args();r=probe(a.order);(ROOT/f'certs/sos_exact_native_m{a.order}.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['gram_matrices','basis_scales']},flush=True)
