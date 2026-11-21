import sys
from an4 import *
name=sys.argv[1]; g=loadg(name,400,128); C=load_cands(name)
C=[c for c in C if min(abs(float(x.imag)) for x in c['xs'])<0.1]
print(name,len(C),'principal-like cands;  n, P_n=n!g_n exact, P_n predicted(all), sign')
for n in list(range(4,46,2))+[60,100]:
    p=gpred(C,n); p=p.real if hasattr(p,'real') else p; print('%3d %11.3e %11.3e %s%s'%(n,mp.factorial(n)*g[n],mp.factorial(n)*p,'+' if g[n]>0 else '-','+' if p>0 else '-'))
