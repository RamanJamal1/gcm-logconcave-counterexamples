"""Certify additional derivative orders for the same explicit witness.

Usage: python src/witness_order_table.py [orders ...] [--force]
Without orders, every order 6..20 is covered. Existing per-order certificates in
certs/witness_order_table_paired15/ are kept unless --force is given, and
certs/witness_order_table.json is rebuilt from all per-order files.
"""
import os,sys,json,contextlib
from pathlib import Path
from certify_direct_heat import certify
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 args=[a for a in sys.argv[1:] if a!='--force'];force='--force' in sys.argv[1:]
 orders=[int(a) for a in args] or list(range(6,21))
 state=json.loads((ROOT/'checkpoints/research_status.json').read_text());source=state['negative_witness_certificate'];out=ROOT/'certs/witness_order_table_paired15';out.mkdir(exist_ok=True)
 for m in orders:
  path=out/f'm{m}.json'
  if path.exists() and not force:continue
  with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):r=certify(str(ROOT/source),order_override=m)
  r['convexity_certificate']=source
  path.write_text(json.dumps(r,indent=2)+'\n',newline='\n');print(m,r['signed_derivative_interval'],flush=True)
 rows=[]
 for path in sorted(out.glob('m*.json'),key=lambda p:int(p.stem[1:])):
  r=json.loads(path.read_text())
  rows.append({'order':r['order'],'signed_derivative_interval':r['signed_derivative_interval'],'negative':r['certified_negative_derivative'],'certificate':path.relative_to(ROOT).as_posix()})
 (ROOT/'certs/witness_order_table.json').write_text(json.dumps(rows,indent=2)+'\n',newline='\n')
 print('table orders:',[r['order'] for r in rows])
