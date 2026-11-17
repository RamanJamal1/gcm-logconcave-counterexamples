"""Illustrative plots; the interval certificates, not the sampled plot, prove signs."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,matplotlib.pyplot as plt
import json
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'checkpoints/research_status.json').read_text());cert=json.loads((ROOT/state['negative_witness_certificate']).read_text());curv=json.loads((ROOT/state['curvature_certificate']).read_text())
x=np.linspace(-20,20,4001);a=np.array([float(Fraction(v)) for v in cert['parameters']['amplitudes']]);f=np.arange(1,len(a)+1);k=float(Fraction(cert['parameters']['kappa']))
cos=np.cos(f[:,None]*x);sin=np.sin(f[:,None]*x);g=1+a@cos;gp=-(a*f)@sin;gpp=-(a*f*f)@cos;xi=1+sum(a*np.exp(-f*f/(2*k)));curvature=k-gpp/g+(gp/g)**2
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans'})
fig,axs=plt.subplots(2,1,figsize=(6.5,4.7),sharex=True,layout='constrained')
axs[0].plot(x,g/xi,color='#184e77',lw=1.5);axs[0].axhline(1,color='0.55',lw=.8,ls=':');axs[0].set_ylabel(r'$p(x)/\varphi_{1/\kappa}(x)$');axs[0].set_title('Certified order-16 witness: density ratio and curvature',loc='left',fontsize=11)
axs[1].plot(x,curvature,color='#184e77',lw=1.5,label=r'$V^{\prime\prime}(x)$ (sampled)');axs[1].axhline(float(Fraction(curv['minimum_curvature_exact'])),color='#c65b16',lw=1.1,ls='--',label='Exact global minimum');axs[1].set_ylabel(r'$V^{\prime\prime}(x)$');axs[1].set_xlabel(r'$x$');axs[1].legend(frameon=False,loc='upper right',fontsize=8);axs[1].set_ylim(-.004,.135)
for ax in axs:ax.grid(alpha=.15);ax.set_xlim(-20,20)
for ext in ['pdf','png']:fig.savefig(ROOT/f'paper/witness16.{ext}',dpi=180)
print('saved paper/witness16.pdf and paper/witness16.png')
