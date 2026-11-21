import torch,mpmath as mp,sys,time,random
sys.path.insert(0,"."); import dd,ddla
mp.mp.prec=400;n=int(sys.argv[1]);nb=int(sys.argv[2])
random.seed(1)
Qm=mp.matrix(n,n)
for i in range(n):
    for j in range(n):Qm[i,j]=mp.mpf(random.gauss(0,1))
Q,_=mp.qr(Qm)
lam=[mp.mpf(10)**(-20*i/(n-1)) for i in range(n)]
M=Q*mp.diag(lam)*Q.T
def todd(x):
    h=float(x);return h,float(x-mp.mpf(h))
Mh=torch.zeros(n,n,dtype=torch.float64);Ml=torch.zeros_like(Mh)
for i in range(n):
    for j in range(i,n):
        h,l=todd(M[i,j]);Mh[i,j]=Mh[j,i]=h;Ml[i,j]=Ml[j,i]=l
Mr=mp.matrix(n,n)
for i in range(n):
    for j in range(n):Mr[i,j]=mp.mpf(float(Mh[i,j]))+mp.mpf(float(Ml[i,j]))
bv=mp.matrix([mp.mpf(random.gauss(0,1)) for _ in range(n)])
bh=torch.tensor([float(v) for v in bv],dtype=torch.float64);bl=torch.tensor([float(v-mp.mpf(float(v))) for v in bv],dtype=torch.float64)
xref=mp.lu_solve(Mr,bv)
M_=(Mh.cuda(),Ml.cuda());t=time.time();L=ddla.cholesky(M_,nb);torch.cuda.synchronize();print("chol time",time.time()-t)
r=(bh.cuda(),bl.cuda());t=time.time();x=ddla.chol_solve(L,r,nb);print("solve time",time.time()-t)
xm=mp.matrix([mp.mpf(float(a))+mp.mpf(float(b)) for a,b in zip(x[0].cpu(),x[1].cpu())])
print("fwd err rel",float(mp.norm(xm-xref)/mp.norm(xref)))
# refinement
res=bv-Mr*xm
rh=torch.tensor([float(v) for v in res],dtype=torch.float64).cuda();rl=torch.tensor([float(v-mp.mpf(float(v))) for v in res],dtype=torch.float64).cuda()
c=ddla.chol_solve(L,(rh,rl),nb);cm=mp.matrix([mp.mpf(float(a))+mp.mpf(float(b)) for a,b in zip(c[0].cpu(),c[1].cpu())])
print("after 1 refine err rel",float(mp.norm(xm+cm-xref)/mp.norm(xref)))
