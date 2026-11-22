import numpy as np, scipy.sparse as sp, sdpap
from sdpap import SymCone
# min x1+x2... : min <I,X> s.t. X12 = 1 (X in S^2_+): optimum 2 (X=[[1,1],[1,1]])
# SDPAP CLP: min c^T x, A x = b, x in K ; x=vec(X) (4 entries)
c=sp.csc_matrix(np.array([[1.],[0],[0],[1.]]))
A=sp.csr_matrix(np.array([[0,1.,0,0]])); b=sp.csc_matrix(np.array([[1.]]))
K=SymCone(s=(2,)); J=SymCone(f=1)
x,y,i,t,s=sdpap.solve(A,b,c,K,J,{'print':'no'}) if False else sdpap.solve(A,b,c,K,J)
print('obj',(c.T@x).toarray() if sp.issparse(x) else c.T@x, i)
