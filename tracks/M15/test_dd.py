import torch,mpmath as mp,random,sys,time
sys.path.insert(0,"."); import dd
mp.mp.prec=300
def rnd(shape,rng=0):
    # random DD numbers with mild dynamic range
    h=torch.randn(*shape,dtype=torch.float64,device="cuda")*torch.exp2(torch.randint(-rng,rng+1,shape,device="cuda").double())
    l=torch.randn(*shape,dtype=torch.float64,device="cuda")*h.abs()*2.0**-54
    h2=h+l;l2=l-(h2-h);return (h2,l2)
def tomp(x):
    h=x[0].cpu().tolist();l=x[1].cpu().tolist()
    def f(a):
        return [f(v) for v in a] if isinstance(a,list) else None
    return h,l
def val(h,l):return mp.mpf(h)+mp.mpf(l)
def err_elem(res,ref):
    h=res[0].flatten().cpu().tolist();l=res[1].flatten().cpu().tolist()
    return max(abs((val(a,b)-r)/r) for a,b,r in zip(h,l,ref))
N=200
x=rnd((N,));y=rnd((N,))
xs=[val(a,b) for a,b in zip(x[0].cpu().tolist(),x[1].cpu().tolist())];ys=[val(a,b) for a,b in zip(y[0].cpu().tolist(),y[1].cpu().tolist())]
print("add",float(err_elem(dd.add(x,y),[a+b for a,b in zip(xs,ys)])))
print("mul",float(err_elem(dd.mul(x,y),[a*b for a,b in zip(xs,ys)])))
print("div",float(err_elem(dd.div(x,y),[a/b for a,b in zip(xs,ys)])))
xp=(x[0].abs(),x[1]*torch.sign(x[0]));xps=[abs(a) for a in xs]
print("sqrt",float(err_elem(dd.sqrt(xp),[mp.sqrt(a) for a in xps])))
for (m,k,n,rng) in [(30,40,20,0),(30,40,20,20),(40,708,5,10),(20,5000,3,10)]:
    A=rnd((m,k),rng);B=rnd((k,n),rng)
    t=time.time();C=dd.matmul(A,B);torch.cuda.synchronize();t=time.time()-t
    Ah=A[0].cpu().tolist();Al=A[1].cpu().tolist();Bh=B[0].cpu().tolist();Bl=B[1].cpu().tolist()
    Ch=C[0].cpu().tolist();Cl=C[1].cpu().tolist();worst=0
    for i in range(min(m,6)):
        for j in range(n):
            ref=sum(val(Ah[i][p],Al[i][p])*val(Bh[p][j],Bl[p][j]) for p in range(k))
            mag=sum(abs(val(Ah[i][p],Al[i][p])*val(Bh[p][j],Bl[p][j])) for p in range(k))
            worst=max(worst,abs(val(Ch[i][j],Cl[i][j])-ref)/mag)
    print("matmul",m,k,n,"range",rng,"max err rel to sum|a||b|",float(worst),"time",round(t,3))
