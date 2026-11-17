"""FP64 GPU refinement of analytically positive squared-Fourier curvatures.

V''=epsilon+f(x)^2, normalized to average curvature one. This avoids the
sum-of-absolute-amplitudes restriction of the earlier potential family.
Numerical quadrature results are candidates, not certificates.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,math,time,json,argparse
from pathlib import Path
from generated_extended_integrands import INTEGRANDS
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def decode(raw,K):
 eps=.0001+.3*torch.sigmoid(raw[:,0]);omega=.15+3.85*torch.sigmoid(raw[:,1]);f=raw[:,2:];energy=f[:,0]**2+.5*(f[:,1:]**2).sum(1);f=f*torch.sqrt((1-eps)/energy.clamp_min(1e-16))[:,None]
 c=[(f[:,j]-1j*f[:,K+j])/2 for j in range(1,K+1)];coeff=[x.conj() for x in c[::-1]]+[f[:,0].to(torch.complex128)]+c
 d=[]
 for n in range(1,2*K+1):d.append(sum((coeff[i]*coeff[n+2*K-i] for i in range(2*K+1) if 0<=n+2*K-i<=2*K),torch.zeros_like(coeff[0])))
 d=torch.stack(d,1);return eps,omega,f,2*d.real,-2*d.imag

def evaluate(raw,K,m,grid=2049,extent=16):
 eps,omega,f,A,B=decode(raw,K);freq=omega[:,None]*torch.arange(1,2*K+1,device=raw.device,dtype=raw.dtype)[None,:]
 # Mode centering is detached; the quadrature itself remains differentiable.
 with torch.no_grad():
  bound=((abs(A)+abs(B))/freq).sum(1)+1;lo=-bound;hi=bound
  for _ in range(35):
   mode=(lo+hi)/2;th=freq*mode[:,None];vp=mode+((A*torch.sin(th)-B*torch.cos(th))/freq).sum(1);lo=torch.where(vp<0,mode,lo);hi=torch.where(vp<0,hi,mode)
  mode=(lo+hi)/2
 x=mode[:,None]+torch.linspace(-extent,extent,grid,device=raw.device,dtype=raw.dtype)[None,:];th=freq[:,:,None]*x[:,None,:];co=torch.cos(th);si=torch.sin(th);cycleco=(co,-si,-co,si);cyclesi=(si,co,-si,-co)
 lp=-x*x/2+((A[:,:,None]*co+B[:,:,None]*si)/freq[:,:,None]**2).sum(1);u={}
 for j in range(2,m+1):u[j]=-((A[:,:,None]*cycleco[(j-2)%4]+B[:,:,None]*cyclesi[(j-2)%4])*freq[:,:,None]**(j-2)).sum(1)-(1 if j==2 else 0)
 density=torch.exp(lp-lp.max(1,keepdim=True).values);density=density/torch.trapezoid(density,x,dim=1)[:,None];mean=torch.trapezoid(x*density,x,dim=1);var=torch.trapezoid((x-mean[:,None])**2*density,x,dim=1);signed=(-1)**m*torch.trapezoid(INTEGRANDS[m](u)*density,x,dim=1);return signed*var**m,signed,var

def search(m,K,batch,steps,seed):
 torch.set_num_threads(1);torch.manual_seed(9123+seed);start=time.time();raw=torch.randn(batch,2+2*K+1,device='cuda',dtype=torch.float64);raw[:,0]-=2;raw[:,1]=torch.linspace(-2,1.5,batch,device='cuda');raw[:,2]+=1;raw[:,3:]*=.3;raw.requires_grad_();calls=0
 # Keep a Gaussian control to catch implementation errors.
 control=raw[:1].detach().clone();control[:,2]=1;control[:,3:]=0
 with torch.no_grad():assert abs(float(evaluate(control,K,m,1025)[0])-math.factorial(m-1)/2)<1e-9*math.factorial(m-1)
 opt=torch.optim.LBFGS([raw],max_iter=steps,max_eval=steps*2,history_size=15,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-12)
 def closure():
  nonlocal calls
  opt.zero_grad();v=evaluate(raw,K,m,2049)[0];loss=torch.asinh(v/math.factorial(m-1)).mean()
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite squared-curvature objective')
  loss.backward();calls+=1
  if calls%10==0:print(m,calls,float(v.detach().min()),time.time()-start,flush=True)
  return loss
 opt.step(closure)
 with torch.no_grad():
  coarse=evaluate(raw,K,m,4097)[0];fine,signed,var=evaluate(raw,K,m,8193,20);valid=torch.isfinite(fine)&(abs(coarse-fine)<1e-5*torch.maximum(abs(fine),torch.ones_like(fine)));idx=int(torch.where(valid,fine,torch.full_like(fine,float('inf'))).argmin());eps,w,f,A,B=decode(raw,K);r={'order':m,'status':'bounded_run_complete','certified':False,'family':'epsilon_plus_squared_fourier_curvature','raw_parameters':raw[idx].cpu().tolist(),'epsilon':float(eps[idx]),'omega':float(w[idx]),'square_coefficients':f[idx].cpu().tolist(),'curvature_cos':A[idx].cpu().tolist(),'curvature_sin':B[idx].cpu().tolist(),'normalized':float(fine[idx]),'signed':float(signed[idx]),'variance':float(var[idx]),'resolution_valid':bool(valid[idx]),'coarse_fine_difference':float(abs(coarse[idx]-fine[idx])),'refinement_candidates':batch,'calls':calls,'seconds':time.time()-start};write_json(ROOT/f'checkpoints/squared_curvature_m{m}_s{seed}.json',r);print(r,flush=True)
 return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--order',type=int,default=15);p.add_argument('--terms',type=int,default=3);p.add_argument('--batch',type=int,default=24);p.add_argument('--steps',type=int,default=100);p.add_argument('--seed',type=int,default=0);a=p.parse_args();search(a.order,a.terms,a.batch,a.steps,a.seed)
