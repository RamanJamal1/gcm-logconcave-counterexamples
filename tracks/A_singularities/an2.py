import json,sys
from ts import *
mp.mp.dps=50
def loadg(n,N=120,g=256):
    h=[mp.mpf(x) for x in json.load(open("data/hn_%s_N%d_g%d.json"%(n,N,g)))["h"]]
    return [(-1)**k*h[k] for k in range(len(h))]   # g_n = (-1)^n h_n  (repo convention: Taylor coeffs in sigma=-tau)
def S_g(g,kappa,m): return S_from_h(g,kappa,m)
if __name__=="__main__":
    for n in ["witness3","two_mode","R15opt5","zou32"]:
        g=loadg(n); k=CONFIGS[n]["kappa"]
        S=[S_g(g,k,m) for m in range(0,41)]
        print(n,'S_m m=12..20:',[mp.nstr(S[m],5) for m in range(12,21)])
        neg=[m for m in range(1,41) if S[m]<0]; print('  negative orders<=40:',neg)
        print('  neg P_n=n!g_n n<=60:',[m for m in range(1,61) if g[m]<0])
        json.dump({'S':[mp.nstr(x,20) for x in S]},open('data/S_%s.json'%n,'w'))
