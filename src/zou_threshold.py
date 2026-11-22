"""Finite-order signs for the Zou-Fan-Gao-Wang family (arXiv 2608.30275).

f_M ∝ exp(-kappa x^2/2)(1 + a cos x + b cos 2x), a = 2^(-M/4), b = 2^(-M), kappa = 8a.
Their paper proves (-1)^M H^(M) < 0 only for "sufficiently large M". Here every
member f_M, 6 <= M <= 400, is evaluated at its own order M with the rigorous Arb
series certifier (src/certify_fourier_series.certify_balls). The parameters are the
exact ones: a is an Arb ball containing 2^(-M/4), b = 2^(-M) is exact, and
kappa = 8a is the corresponding ball. No rational stand-ins are used and no member
is skipped.

Usage: python src/zou_threshold.py [first_M last_M [output.json]]
"""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from flint import arb,ctx
from certify_fourier_series import certify_balls

SETTINGS=[dict(N=160,dps=120,mode_cutoff=16),dict(N=240,dps=200,mode_cutoff=24)]

def evaluate(M):
    for settings in SETTINGS:
        ctx.dps=settings['dps']
        a=arb(2)**(-arb(M)/4);b=arb(2)**(-M);kappa=8*a
        t=time.time()
        r=certify_balls(M,a,b,kappa,parameters={'a':'2^(-%d/4)'%M,'b':'2^(-%d)'%M,'kappa':'8*2^(-%d/4)'%M},**settings)
        value=arb(r['signed_derivative_interval'])
        sign='negative' if value<0 else 'positive' if value>0 else 'undecided'
        if sign!='undecided':break
    return {'order':M,'a_ball':str(a),'b':'2^(-%d)'%M,'kappa_ball':str(kappa),'exact_member':True,'sign':sign,
            'signed_derivative_interval':r['signed_derivative_interval'],
            'normalized_signed_interval':r['normalized_signed_interval'],
            'settings':settings,'seconds':round(time.time()-t,2)}

def main():
    first,last=(int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else (6,400)
    out=Path(sys.argv[3]) if len(sys.argv)>3 else Path('checkpoints/zou_family_threshold.json')
    start=time.time();rows=[]
    for M in range(first,last+1):
        rows.append(evaluate(M));print(json.dumps({k:rows[-1][k] for k in ('order','sign','seconds')}),flush=True)
    pos=[r['order'] for r in rows if r['sign']=='positive'];neg=[r['order'] for r in rows if r['sign']=='negative']
    und=[r['order'] for r in rows if r['sign']=='undecided']
    summary={'positive_orders':pos,'negative_orders':neg,'undecided':und,
             'first_negative_order':min(neg) if neg else None,'last_positive_order':max(pos) if pos else None,
             'all_members_exact':True,'total_seconds':round(time.time()-start,1),
             'note':'Rigorous Arb enclosures for the exact family members: a = 2^(-M/4) is enclosed in an Arb ball, '
                    'b = 2^(-M) is exact and kappa = 8a. Settings escalate to N=240, dps=200, mode_cutoff=24 only if '
                    'the sign is undecided at N=160, dps=120, mode_cutoff=16.'}
    out.write_text(json.dumps({'family':'Zou-Fan-Gao-Wang f_M (arXiv 2608.30275), member f_M evaluated at order M',
                               'method':'src/certify_fourier_series.certify_balls with exact Arb-ball parameters',
                               'summary':summary,'rows':rows},indent=2)+'\n',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('positive_orders','negative_orders')}))

if __name__=='__main__':
    main()
