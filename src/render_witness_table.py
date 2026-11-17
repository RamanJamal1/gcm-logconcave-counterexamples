"""Render rounded midpoints only after checking the stored interval signs."""
import json
from pathlib import Path
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]
ctx.dps=200
rows=json.loads((ROOT/'certs/witness_order_table.json').read_text());lines=[r'\begin{table}[htbp]',r'\centering',r'\begin{tabular}{rrc}',r'\toprule',r'$m$ & Approximate $S_m(p)$ & Certified sign \\',r'\midrule']
for row in rows:
 ball=arb(row['signed_derivative_interval']);assert ball>0 or ball<0
 lines.append(f"{row['order']} & ${float(ball.mid()):.8g}$ & "+('positive' if ball>0 else 'negative')+r' \\')
folder=str(Path(rows[0]['certificate']).parent)
lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Derivatives of the same explicit order-16 witness. Displayed values are rounded midpoints; signs are rigorously certified. Full enclosures are in \path{'+folder+r'/}. These are statements about this density only.}',r'\end{table}']
(ROOT/'paper/witness_table.tex').write_text('\n'.join(lines)+'\n')
