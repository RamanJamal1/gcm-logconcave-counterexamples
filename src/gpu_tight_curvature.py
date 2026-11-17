"""FP64 batched optimization of signed Fourier harmonics at fixed amplitudes.

A_j = a * j^(-m) * x_j scales the leading periodic quadratic term to
approximately equal curvature. Outputs are screening candidates only.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,math,time
from pathlib import Path
import torch
from periodic_cuda import evaluate

def search(m,K,batch=96,steps=120,seed=0):
 torch.set_num_threads(1);torch.manual_seed(915+seed);start=time.time();device='cuda';dtype=torch.float64
 a=torch.exp(torch.linspace(math.log(.008),math.log(.6),batch,device=device,dtype=dtype));j=torch.arange(2,K+1,device=device,dtype=dtype)
 h=torch.tensor([.5 if w==2 else (-1)**w*.1**(w-2) for w in range(2,K+1)],device=device,dtype=dtype)
 initial=a[:,None]**(j-1)*((j+1)/2)**m*h
 if seed:initial*=torch.exp(torch.randn_like(initial)*.3)
 raw=torch.stack([initial,torch.randn_like(initial)*.1]).requires_grad_();frequencies=torch.arange(1,K+1,device=device,dtype=dtype)
 def decode():
  amps=torch.cat([a[:,None],a[:,None]*j**(-m)*raw[0]],dim=1);sine=torch.cat([torch.zeros_like(a[:,None]),a[:,None]*j**(-m)*raw[1]],dim=1);magnitudes=torch.sqrt(amps**2+sine**2);A=magnitudes.sum(1);theta=torch.arange(1024,device=device,dtype=dtype)*2*math.pi/1024;co=torch.cos(frequencies[:,None]*theta);si=torch.sin(frequencies[:,None]*theta)
  g=1+amps@co+sine@si;gp=-(amps*frequencies)@si+(sine*frequencies)@co;gpp=-(amps*frequencies**2)@co-(sine*frequencies**2)@si
  moments=[(magnitudes*frequencies**r).sum(1)/(1-A) for r in range(1,5)];q1,q2,q3,q4=moments;bound4=q4+4*q1*q3+3*q2*q2+12*q1*q1*q2+6*q1**4
  k=(gpp/g-(gp/g)**2).max(1).values+(2*math.pi/1024)**2*bound4/8+1e-7
  return amps,sine,k
 opt=torch.optim.LBFGS([raw],lr=1,max_iter=steps,max_eval=steps*2,history_size=30,line_search_fn='strong_wolfe',tolerance_grad=1e-11,tolerance_change=1e-13);calls=0
 def closure():
  nonlocal calls
  opt.zero_grad();amps,sine,k=decode();score=evaluate(amps,k,m,512,sine)/(a*a);loss=torch.asinh(score).mean();loss.backward();calls+=1
  if calls%20==0:print('order',m,'calls',calls,'min_screen',float(score.detach().min()),'seconds',time.time()-start,flush=True)
  return loss
 opt.step(closure)
 with torch.no_grad():
  amps,sine,k=decode();coarse=evaluate(amps,k,m,1024,sine);fine=evaluate(amps,k,m,2048,sine);valid=torch.isfinite(fine)&(abs(fine-coarse)<1e-7*torch.maximum(torch.ones_like(fine),abs(fine)))&(abs(amps).sum(1)<1)&(k>0)
  scores=torch.where(valid,fine/(a*a),torch.full_like(fine,float('inf')));index=int(scores.argmin());r={'order':m,'terms':K,'amplitudes':amps[index].cpu().tolist(),'sine_amplitudes':sine[index].cpu().tolist(),'kappa':float(k[index]),'periodic_surrogate':float(fine[index]),'normalized_screen':float(scores[index]),'coarse_fine_error':float(abs(fine[index]-coarse[index])),'certified':False,'calls':calls,'refinement_candidates':batch,'seconds':time.time()-start,'curvature_method':'grid maximum plus global fourth-log-derivative interpolation bound (floating screening)', 'note':'Periodic surrogate with full Gaussian corrections omitted; requires spatial certificate'}
 Path(f'checkpoints/gpu_tight_curvature_K{K}_m{m}_s{seed}.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',default=[15]);p.add_argument('--terms',type=int,default=6);p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 for m in a.orders:search(m,a.terms,steps=a.steps,seed=a.seed)
