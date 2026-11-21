"""Dual-ray screen: d = y_new - y_old. Ray iff mat(A^T d) <= 0 (both blocks) and b^T d > 0. float64 screen only.
Usage: python3 raytest.py old.npz new.npz"""
import sys,numpy as np,scipy.sparse as sp
D=np.load('data/dd_m15.npz');dims=[int(v) for v in D['dims']];N=sum(n*n for n in dims);R=len(D['bh'])
A=sp.csr_matrix((D['H'],(D['R'],D['C'])),shape=(R,N));b=D['bh']
o,n=np.load(sys.argv[1]),np.load(sys.argv[2])
yo=o['yh']+o['yl'];yn=n['yh']+n['yl'];d=yn-yo
print('|y_old|',np.linalg.norm(yo),'|y_new|',np.linalg.norm(yn),'|d|',np.linalg.norm(d),'cos(d,y)',d@yn/np.linalg.norm(d)/np.linalg.norm(yn))
v=A.T@d;off=0
for k,m in enumerate(dims):
    M=v[off:off+m*m].reshape(m,m);M=(M+M.T)/2;off+=m*m;e=np.linalg.eigvalsh(M)
    print(f'block{k}: eig max {e[-1]:.3e} min {e[0]:.3e}  #pos>{1e-8*abs(e).max():.1e}: {(e>1e-8*abs(e).max()).sum()}')
print('b.d',b@d,' b.d/|d|',b@d/np.linalg.norm(d))
