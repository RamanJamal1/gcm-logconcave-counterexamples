"""Update the proved order bracket only after independent certificate verification."""
import argparse,json,hashlib,re,time
from pathlib import Path
from verify_sos_standalone import verify
ROOT=Path(__file__).resolve().parents[1]
def update(orders):
 path=ROOT/'checkpoints/research_status.json'
 state=json.loads(path.read_text()) if path.exists() else {'nonnegative_orders':[6,7,8],'sharp_gaussian_orders':[6,7,8],'negative_witness_order':17,'negative_witness_certificate':'certs/simple_fourier_m17.json','verified_certificates':{}}
 for m in orders:
  cert=ROOT/f'certs/exact_quotient_m{m}.json';result=verify(cert);result['sha256']=hashlib.sha256(cert.read_bytes()).hexdigest();result['certificate']=str(cert.relative_to(ROOT));state['verified_certificates'][str(m)]=result
  if m not in state['nonnegative_orders']:state['nonnegative_orders'].append(m)
 lower=6
 while lower in state['nonnegative_orders']:lower+=1
 state.update(lower_bound=lower,upper_bound=state['negative_witness_order'],updated=time.time(),exact_minimal_order_resolved=lower==state['negative_witness_order'])
 state['nonnegative_orders'].sort();path.write_text(json.dumps(state,indent=2)+'\n')
 p=ROOT/'MORNING_BRIEFING.md';s=p.read_text();s=re.sub(r'\*\*\d+ ≤ m\* ≤ \d+\*\*',f'**{lower} ≤ m* ≤ {state["upper_bound"]}**',s)
 s=re.sub(r'A further exact rational certificate proves.*?An explicit strictly',f'Additional exact rational certificates prove nonnegative signs through order {lower-1} (`certs/exact_quotient_m*.json`), independently verified using standard-library rational arithmetic. An explicit strictly',s);p.write_text(s)
 p=ROOT/'paper/analytic_scope.tex';s=p.read_text();s=re.sub(r'For \$m\\in\\\{[0-9,]+\\\}\$',lambda match:'For $m\\in\\{'+','.join(map(str,range(6,lower)))+'\\}$',s)
 s=re.sub(r'The first possible failing order is at least \w+;',f'The first possible failing order is at least {lower};',s)
 s=s.replace('orders six through nine',f'orders six through {lower-1}')
 # Keep a separate explicit audit paragraph rather than inferring the scope from file names.
 marker='\\paragraph{Additional verified orders.}'
 if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
 s+=f'\n{marker}\nExact rational quotient certificates and independent verification now cover orders '+', '.join(map(str,sorted(state['verified_certificates'].keys(),key=int)))+'. The machine-readable verification records and certificate hashes are stored in \\texttt{checkpoints/research\\_status.json}.\n'
 p.write_text(s)
 p=ROOT/'paper/explicit_upper_bound.tex';s=p.read_text();s=re.sub(r'between orders [a-z0-9]+ and 17',f'between orders {lower} and 17',s);p.write_text(s)
 print(json.dumps({'lower_bound':lower,'upper_bound':state['upper_bound'],'verified_orders':orders}),flush=True)
 return state
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--orders',type=int,nargs='+',required=True);a=p.parse_args();update(a.orders)
