"""Certify additional derivative orders for the same explicit witness."""
import os,json,contextlib
from pathlib import Path
from certify_direct_heat import certify
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 out=ROOT/'certs/witness_order_table';out.mkdir(exist_ok=True);rows=[]
 for m in [6,7,8,12,13,14,15,16,17,18,19,20]:
  with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):r=certify('certs/multifourier_m16.json',order_override=m)
  (out/f'm{m}.json').write_text(json.dumps(r,indent=2)+'\n');row={'order':m,'signed_derivative_interval':r['signed_derivative_interval'],'negative':r['certified_negative_derivative'],'certificate':str((out/f'm{m}.json').relative_to(ROOT))};rows.append(row);print(row,flush=True);(ROOT/'certs/witness_order_table.json').write_text(json.dumps(rows,indent=2)+'\n')
