import json
from ts import *
mp.mp.dps=50
def load(n,N=120,g=256): return [mp.mpf(x) for x in json.load(open("data/hn_%s_N%d_g%d.json"%(n,N,g)))["h"]]
if __name__=="__main__":
    h=load("witness3"); k=CONFIGS["witness3"]["kappa"]
    print([ (m,mp.nstr(S_from_h(h,k,m),6)) for m in range(12,21)])
    for n in ["witness3","two_mode","R15opt5","zou32"]:
        h=load(n); k=CONFIGS[n]["kappa"]
        print(n,[ (m,mp.nstr(S_from_h(h,k,m),4)) for m in (6,10,14,15,16,17,20,32,33)])
        print(' h_n/h_{n-1}:',[mp.nstr(h[m]/h[m-1],6) for m in (20,40,60,80,100,120)])
        print(' sign h_n n=1..40:',''.join('+' if h[m]>0 else '-' for m in range(1,41)))
