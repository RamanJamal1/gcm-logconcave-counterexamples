"""Fast exact identity pre-check with fmpq_mpoly (seconds). Independent of the Fraction verifier's code path."""
import ast,json,math,time,sys
from flint import fmpq_mpoly_ctx,fmpq
N=32
ctx=fmpq_mpoly_ctx.get(tuple(f'u{i}' for i in range(1,N+1)),'lex')
G=ctx.gens()
def parse(text):
    def rec(n):
        if isinstance(n,ast.Constant) and isinstance(n.value,int):return ctx.constant(n.value)
        if isinstance(n,ast.Name):return G[int(n.id[1:])-1]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -rec(n.operand)
        if isinstance(n,ast.BinOp):
            if isinstance(n.op,ast.Add):return rec(n.left)+rec(n.right)
            if isinstance(n.op,ast.Sub):return rec(n.left)-rec(n.right)
            if isinstance(n.op,ast.Mult):return rec(n.left)*rec(n.right)
            if isinstance(n.op,ast.Div):
                r=rec(n.right)
                if not r.is_constant():raise ValueError('nonconstant divisor')
                return rec(n.left)*(1/fmpq(int(str(r.leading_coefficient()).split('/')[0]),int(str(r.leading_coefficient()).split('/')[1]) if '/' in str(r.leading_coefficient()) else 1))
            if isinstance(n.op,ast.Pow):
                p=rec(n.left);out=ctx.constant(1)
                for _ in range(n.right.value):out=out*p
                return out
        raise ValueError(ast.dump(n))
    return rec(ast.parse(text,mode='eval').body)
def qconst(txt):
    a=txt.split('/');return ctx.constant(fmpq(int(a[0]),int(a[1]) if len(a)>1 else 1))
def entropy(m):
    f=G[1]*G[1]*fmpq(1,2)
    for _ in range(2,m):
        out=ctx.constant(0)
        for i in range(2,N-1):
            fi=f.derivative(i-1)
            if fi.is_zero():continue
            for j in range(2,N-1):
                d=fi.derivative(j-1)
                if not d.is_zero():out=out+d*G[i]*G[j]*fmpq(-1,2)
            for k in range(1,i):out=out+fi*G[k]*G[i-k]*fmpq(math.comb(i,k),2)
        f=out
    return f*((-1)**m)
def check(path):
    t=time.time();d=json.load(open(path));m=d['order'];tot=ctx.constant(0)
    for b,(base,mat) in enumerate(zip(d['bases'],d['gram_matrices'])):
        bs=[parse(x) for x in base];blk=ctx.constant(0)
        for i,x in enumerate(bs):
            for j,y in enumerate(bs):
                blk=blk+x*y*qconst(mat[i][j])
        tot=tot+(blk if b==0 else -G[1]*blk)
    for e,c in zip(d['ibp_polynomials'],d['ibp_multipliers']):tot=tot-parse(e)*qconst(c)
    diff=tot-entropy(m)
    return {'identity_exact':bool(diff.is_zero()),'seconds':time.time()-t}
if __name__=='__main__':print(check(sys.argv[1]))
