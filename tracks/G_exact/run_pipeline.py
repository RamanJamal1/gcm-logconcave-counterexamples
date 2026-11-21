"""run_pipeline.py m seed.json out_cert.json [--kbits 256] [--tol 160]: project -> PD checks -> cert -> precheck."""
import sys,json,time,argparse,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import exact_project,pd_check,make_cert,precheck
p=argparse.ArgumentParser();p.add_argument('m',type=int);p.add_argument('seed');p.add_argument('out');p.add_argument('--kbits',type=int,default=256);p.add_argument('--tol',type=int,default=160);p.add_argument('--arb',action='store_true');a=p.parse_args()
t0=time.time()
xe,info,S,seed=exact_project.project(a.m,a.seed,a.kbits,a.tol)
grams=exact_project.grams_from_x(S,xe);pd=[]
for Q in grams:
    r=pd_check.rump(Q);print('rump',r,flush=True);pd.append(r)
    if a.arb:
        r2=pd_check.arb_ldl(Q);print('arb',{k:v for k,v in r2.items() if k!='min_pivot'},flush=True);pd.append(r2)
if not all(r['proved_pd'] for r in pd):
    print('PD CHECK FAILED -- not writing certificate');sys.exit(2)
cert,_=make_cert.build(S,xe,a.seed,t0,{'projection':info,'pd_fast_checks':pd})
cert['status']='exact_identity_and_positive_definite_grams_verified'
json.dump(cert,open(a.out,'w'));print('wrote',a.out,os.path.getsize(a.out),'bytes',time.time()-t0,flush=True)
print('precheck',precheck.check(a.out),flush=True)
