"""Supplementary exact-parameter exploration of the explicit two-mode family."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from fractions import Fraction as F
from pathlib import Path
from certify_fourier_series import certify
from checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1]

def main(args):
 state={'evaluations':0,'orders':{},'status':'running','started':time.time(),'method':'Arb finite Fourier series screening; final negatives at 200 digits'}
 for m in args.orders:
  best=None
  for power in range(3,14):
   a=F(1,2**power)
   for coefficient in [F(1,4),F(1,2),F(1),F(2),F(4)]:
    b=coefficient*a*a*F(3,4)**m
    for margin in [F(101,100),F(11,10),F(2)]:
     kappa=margin*(a+4*b)/(1-a-b)
     r=certify(m,str(a),str(b),str(kappa),N=96,dps=100,mode_cutoff=12)
     state['evaluations']+=1
     if best is None or r['normalized_signed_midpoint']<best['normalized_signed_midpoint']:
      best=r;state['orders'][str(m)]=r
      (ROOT/f'checkpoints/series_best_m{m}.json').write_text(json.dumps(r,indent=2)+'\n')
      if args.sync:checkpoint(f'series m{m} best normalized={r["normalized_signed_midpoint"]:.9g}')
     if r['certified']:
      final=certify(m,str(a),str(b),str(kappa),N=128,dps=200,mode_cutoff=16)
      assert final['certified']
      (ROOT/f'certs/series_hit_m{m}.json').write_text(json.dumps(final,indent=2)+'\n')
      state['first_certified_order']=m;state['status']='certified_upper_bound_found'
      (ROOT/'checkpoints/series_campaign.json').write_text(json.dumps(state,indent=2)+'\n')
      if args.sync:checkpoint(f'certified Fourier-family violation at order {m}')
      print('CERTIFIED',m,final['signed_derivative_interval'],flush=True);return
     if state['evaluations']%20==0:
      (ROOT/'checkpoints/series_campaign.json').write_text(json.dumps(state,indent=2)+'\n');print(m,state['evaluations'],best['normalized_signed_midpoint'],flush=True)
  print('order completed',m,best['normalized_signed_midpoint'],flush=True)
 state['status']='finite_sweep_complete';(ROOT/'checkpoints/series_campaign.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=list(range(9,49)));p.add_argument('--sync',action='store_true');main(p.parse_args())
