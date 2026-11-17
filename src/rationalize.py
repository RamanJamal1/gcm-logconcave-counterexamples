"""Round a candidate to exact small fractions; certify every retained variant."""
import argparse,json
from fractions import Fraction
from pathlib import Path
from certify_arb import certify

def rationalize(c,denominator=100):
 c=c.get('parameters',c)
 return {k:([str(Fraction(str(x)).limit_denominator(denominator)) for x in v] if isinstance(v,list) else str(Fraction(str(v)).limit_denominator(denominator)) if k!='family' else v) for k,v in c.items()}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--order',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();c=json.loads(Path(a.candidate).read_text());out=[]
 for denominator in [10,25,50,100]:
  candidate=rationalize(c,denominator)
  try:r=certify(candidate,a.order)
  except Exception as e:r={'status':'rejected','certified':False,'error':str(e),'parameters':candidate}
  out.append({'denominator_limit':denominator,'certificate':r})
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
