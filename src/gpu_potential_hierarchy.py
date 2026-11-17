"""Full Gaussian quadrature for high-order Fourier potentials near a known hit.

The first potential harmonic is fixed separately for each restart. Higher
coefficients are scaled by n^-m, and kappa uses a global interpolation bound.
This is numerical refinement only; interval certification is still required.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np,math,time,argparse
from pathlib import Path
import mpmath as mp
from generated_extended_integrands import INTEGRANDS
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def seed(a,K):
 mp.mp.dps=100;a=mp.mpf(str(a));b=a*a*mp.mpf('0.005625');c=-a**3*mp.mpf('0.0001875');ts=mp.polyroots([4*c,2*b,a-3*c,1-b],maxsteps=1000,extraprec=100);inside=[]
 for t in ts:
  root=mp.sqrt(t*t-1);outside=t+root if abs(t+root)>1 else t-root;inside.append(1/outside)
 assert len(inside)==3 and all(abs(r)<1 for r in inside)
 values=[-2*sum(r**n for r in inside)/n for n in range(1,K+1)];assert max(abs(mp.im(v)) for v in values)<mp.mpf('1e-80');return [float(mp.re(v)) for v in values]
def evaluate(amps,m,grid=2049):
 B,K=amps.shape;n=torch.arange(1,K+1,device=amps.device,dtype=amps.dtype);theta=torch.linspace(-math.pi,math.pi,1025,device=amps.device,dtype=amps.dtype);cos=torch.cos(n[:,None]*theta);k=(-(amps*n*n)@cos).max(1).values+(2*math.pi/1024)**2*(abs(amps)*n**4).sum(1)/8+1e-7
 x=torch.linspace(-12,12,grid,device=amps.device,dtype=amps.dtype)[None,:]/k.detach().sqrt()[:,None];th=n[None,:,None]*x[:,None,:];co=torch.cos(th);si=torch.sin(th);cy=(co,-si,-co,si);lp=-k[:,None]*x*x/2+(amps[:,:,None]*co).sum(1);u={j:(amps[:,:,None]*n[None,:,None]**j*cy[j%4]).sum(1)-(k[:,None] if j==2 else 0) for j in range(2,m+1)};density=torch.exp(lp-lp.max(1,keepdim=True).values);density=density/torch.trapezoid(density,x,dim=1)[:,None];value=(-1)**m*torch.trapezoid(INTEGRANDS[m](u)*density,x,dim=1);return value,k

def search(m,K,batch,steps,seed_id):
 torch.set_num_threads(1);start=time.time();a=np.geomspace(.015,.15,batch);initial=torch.tensor([seed(v,K) for v in a],device='cuda',dtype=torch.float64);first=initial[:,:1].clone();n=torch.arange(2,K+1,device='cuda',dtype=torch.float64);raw=(initial[:,1:]*n**m/first).detach().clone().requires_grad_();calls=0
 opt=torch.optim.LBFGS([raw],max_iter=steps,max_eval=steps*2,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-11,tolerance_change=1e-13)
 def decode():return torch.cat([first,raw*first*n**(-m)],1)
 def closure():
  nonlocal calls
  opt.zero_grad();values,k=evaluate(decode(),m);score=values/first[:,0]**2;loss=torch.asinh(score).mean()
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite potential hierarchy')
  loss.backward();calls+=1
  if calls%10==0:print(m,calls,float(score.detach().min()),time.time()-start,flush=True)
  return loss
 opt.step(closure)
 with torch.no_grad():
  amps=decode();coarse,k=evaluate(amps,m,4097);fine,k=evaluate(amps,m,8193);valid=torch.isfinite(fine)&(abs(coarse-fine)<1e-7*torch.maximum(abs(fine),torch.ones_like(fine)));scores=torch.where(valid,fine/first[:,0]**2,torch.full_like(fine,float('inf')));index=int(scores.argmin());r={'order':m,'terms':K,'status':'bounded_run_complete','certified':False,'family':'fourier_potential_tight_curvature','amplitudes':amps[index].cpu().tolist(),'kappa':float(k[index]),'signed':float(fine[index]),'objective':float(scores[index]),'resolution_valid':bool(valid[index]),'coarse_fine_error':float(abs(coarse[index]-fine[index])),'refinement_candidates':batch,'calls':calls,'seconds':time.time()-start};write_json(ROOT/f'checkpoints/potential_hierarchy_m{m}_s{seed_id}.json',r);print(r,flush=True)
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--terms',type=int,default=16);p.add_argument('--batch',type=int,default=16);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.terms,a.batch,a.steps,a.seed)
