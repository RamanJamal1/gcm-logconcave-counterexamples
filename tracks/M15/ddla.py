"""DD dense linear algebra on GPU: blocked left-looking Cholesky and triangular solves (DD arithmetic)."""
import torch
import dd

def tree_sum(P,dim=-1):
    """DD sum over last dim of DD tensor P=(h,l) (shape [...,k])."""
    h,l=P
    while h.shape[-1]>1:
        k=h.shape[-1]
        if k%2==1:
            z=torch.zeros_like(h[...,:1]);h=torch.cat([h,z],-1);l=torch.cat([l,z],-1);k+=1
        a=(h[...,:k//2],l[...,:k//2]);b=(h[...,k//2:],l[...,k//2:]);h,l=dd.add(a,b)
    return (h[...,0],l[...,0])

def cholesky(M,nb=32):
    Mh,Ml=M;n=Mh.shape[0];Lh=torch.zeros_like(Mh);Ll=torch.zeros_like(Ml)
    for c0 in range(0,n,nb):
        c1=min(c0+nb,n);Sh=Mh[c0:,c0:c1].clone();Sl=Ml[c0:,c0:c1].clone()
        if c0>0:
            Pm=dd.matmul((Lh[c0:,:c0],Ll[c0:,:c0]),(Lh[c0:c1,:c0].t().contiguous(),Ll[c0:c1,:c0].t().contiguous()))
            Sh,Sl=dd.sub((Sh,Sl),Pm)
        for j in range(c1-c0):
            vh=Sh[j:,j];vl=Sl[j:,j]
            if j>0:
                pr=dd.mul((Sh[j:,:j],Sl[j:,:j]),(Sh[j:j+1,:j],Sl[j:j+1,:j]))
                ts=tree_sum(pr);vh,vl=dd.sub((vh,vl),ts)
            d=dd.sqrt((vh[:1],vl[:1]))
            ch,cl=dd.div((vh,vl),(d[0].expand_as(vh),d[1].expand_as(vh)))
            Sh[j:,j]=ch;Sl[j:,j]=cl
        Lh[c0:,c0:c1]=torch.tril(Sh);Ll[c0:,c0:c1]=torch.tril(Sl)
    return (Lh,Ll)

def solve_lower(L,r,nb=32):
    """Solve L y = r (DD vectors)."""
    Lh,Ll=L;n=Lh.shape[0];yh=torch.zeros_like(r[0]);yl=torch.zeros_like(r[1])
    for c0 in range(0,n,nb):
        c1=min(c0+nb,n);bh=r[0][c0:c1].clone();bl=r[1][c0:c1].clone()
        if c0>0:
            pm=dd.matmul((Lh[c0:c1,:c0],Ll[c0:c1,:c0]),(yh[:c0].view(-1,1),yl[:c0].view(-1,1)))
            bh,bl=dd.sub((bh,bl),(pm[0].view(-1),pm[1].view(-1)))
        for j in range(c1-c0):
            vh,vl=bh[j:j+1],bl[j:j+1]
            if j>0:
                pr=dd.mul((Lh[c0+j:c0+j+1,c0:c0+j],Ll[c0+j:c0+j+1,c0:c0+j]),(yh[c0:c0+j].view(1,-1),yl[c0:c0+j].view(1,-1)))
                ts=tree_sum(pr);vh,vl=dd.sub((vh,vl),(ts[0].view(1),ts[1].view(1)))
            q=dd.div((vh,vl),(Lh[c0+j,c0+j].view(1),Ll[c0+j,c0+j].view(1)))
            yh[c0+j]=q[0][0];yl[c0+j]=q[1][0]
    return (yh,yl)

def solve_upper_t(L,r,nb=32):
    """Solve L^T x = r (DD vectors), L lower."""
    Lh,Ll=L;n=Lh.shape[0];xh=torch.zeros_like(r[0]);xl=torch.zeros_like(r[1])
    starts=list(range(0,n,nb))[::-1]
    for c0 in starts:
        c1=min(c0+nb,n);bh=r[0][c0:c1].clone();bl=r[1][c0:c1].clone()
        if c1<n:
            pm=dd.matmul((Lh[c1:,c0:c1].t().contiguous(),Ll[c1:,c0:c1].t().contiguous()),(xh[c1:].view(-1,1),xl[c1:].view(-1,1)))
            bh,bl=dd.sub((bh,bl),(pm[0].view(-1),pm[1].view(-1)))
        for j in range(c1-c0-1,-1,-1):
            vh,vl=bh[j:j+1],bl[j:j+1]
            if j<c1-c0-1:
                pr=dd.mul((Lh[c0+j+1:c1,c0+j].view(1,-1),Ll[c0+j+1:c1,c0+j].view(1,-1)),(xh[c0+j+1:c1].view(1,-1),xl[c0+j+1:c1].view(1,-1)))
                ts=tree_sum(pr);vh,vl=dd.sub((vh,vl),(ts[0].view(1),ts[1].view(1)))
            q=dd.div((vh,vl),(Lh[c0+j,c0+j].view(1),Ll[c0+j,c0+j].view(1)))
            xh[c0+j]=q[0][0];xl[c0+j]=q[1][0]
    return (xh,xl)

def chol_solve(L,r,nb=32):return solve_upper_t(L,solve_lower(L,r,nb),nb)
