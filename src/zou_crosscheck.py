"""Cross-check the Zou-Fan-Gao-Wang threshold with a second, independent method.

For M = 32, 33, 36 the member f_M is evaluated at order M with the heat-evolution
certifier of src/certify_direct_heat.py (the second method for the order-16 witness),
and compared with the Fourier-series enclosures in checkpoints/zou_family_threshold.json.
certify_direct_heat takes rational parameters: a = 2^(-M/4) is exact for M = 32, 36;
for M = 33 a rational within relative distance 1e-30 of 2^(-33/4) is used.

Usage, from the repository root: python src/zou_crosscheck.py
"""
import json,sys,tempfile
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from flint import arb,ctx
from certify_direct_heat import certify

ROOT=Path(__file__).resolve().parent.parent

def rational_a(M):
    if M%4==0:return Fraction(1,2**(M//4))
    ctx.dps=60;a=arb(2)**(-arb(M)/4)
    q=Fraction(a.mid().str(50,radius=False)).limit_denominator(10**40)
    if not abs(arb(q.numerator)/q.denominator-a)<a*arb('1e-30'):raise ArithmeticError('approximation too coarse')
    return q

def main():
    series={r['order']:r for r in json.loads((ROOT/'checkpoints/zou_family_threshold.json').read_text())['rows']}
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        for M in (32,33,36):
            a=rational_a(M);b=Fraction(1,2**M)
            src=Path(tmp)/f'zou_{M}.json'
            src.write_text(json.dumps({'order':M,'parameters':{'a':str(a),'b':str(b),'kappa':str(8*a)}}))
            r=certify(str(src),L=300,dps=120)
            ctx.dps=130
            heat=arb(r['signed_derivative_interval']);fourier=arb(series[M]['signed_derivative_interval'])
            rows.append({'order':M,'a':str(a),'a_exact':M%4==0,'heat_evolution_interval':r['signed_derivative_interval'],
                         'fourier_series_interval':series[M]['signed_derivative_interval'],
                         'same_sign':bool((heat<0 and fourier<0) or (heat>0 and fourier>0)),
                         'numerator_tail_bound':r['numerator_tail_bound'],'extent':300,'digits':120})
            print(M,r['signed_derivative_interval'],flush=True)
    out=ROOT/'checkpoints/zou_crosscheck.json'
    out.write_text(json.dumps({'method':'src/certify_direct_heat.certify (L=300, dps=120) vs checkpoints/zou_family_threshold.json','rows':rows},indent=2)+'\n',newline='\n')
    return 0 if all(r['same_sign'] for r in rows) else 1

if __name__=='__main__':
    sys.exit(main())
