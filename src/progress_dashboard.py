"""Aggregate actual saved counters; no inference of proof from search outcomes."""
import json,datetime
from pathlib import Path
from checkpoint_io import write_text
ROOT=Path(__file__).resolve().parents[1]
CAMPAIGNS=['campaign_A','campaign_B','campaign_C','higher_campaign','polynomial_campaign','polynomial_wide_campaign','harmonic_refinement_campaign']
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def update():
 research=read(ROOT/'checkpoints/research_status.json');lines=['# GCM research progress','',f'Updated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}','',f"Verified bracket: {research.get('lower_bound','?')} ≤ m* ≤ {research.get('upper_bound','?')}. Exact minimality remains unresolved.",'','| Campaign | Saved status | FP32 candidates | FP64 refined candidates |','|---|---|---:|---:|'];totals=[0,0]
 for name in CAMPAIGNS:
  s=read(ROOT/f'checkpoints/{name}.json')
  if not s:continue
  a=s.get('gpu_evaluations',0);b=s.get('refinement_evaluations',0);totals[0]+=a;totals[1]+=b;lines.append(f"| {name} | {s.get('status','unknown')} | {a:,} | {b:,} |")
 lines+=['',f'Recorded production totals: {totals[0]:,} FP32 candidate evaluations; {totals[1]:,} FP64 refinements. Warmups and readiness runs are excluded. Interrupted partial generations may not be counted.','','Sharp proven normalized bounds: S6 Var^6 ≥ 60; S7 Var^7 ≥ 360; S8 Var^8 ≥ 2520.','Exact rational nonnegativity and sharp Gaussian-bound certificates cover orders 6–11.','An order-16 witness has a spatial Arb certificate; the earlier order-17 witness has independent Fourier-series and spatial certificates.','See MORNING_BRIEFING.md, checkpoints/research_status.json, and paper/draft_note.tex.']
 lines+=['','## Additional work in progress','','Numerical search values and SDP iterates in this table are not proofs.','','| Attempt | Status | Recorded work |','|---|---|---|']
 for label,name in [('Order-12 high-precision SDP','sdpa_segments_m12'),('Order-15 two base frequencies','two_fundamentals_m15_s0'),('Order-15 positive curvature bumps','rbf_higher_m15_s0'),('Order-12 curvature lattice','rbf_lattice_m12_s0')]:
  record=read(ROOT/f'checkpoints/{name}.json')
  if record:
   detail=f"segment {record['segment']}; affine residual {record.get('last_residual',float('nan')):.3g}" if 'segment' in record else f"{record.get('evaluations',0):,} CPU objective evaluations"
   lines.append(f"| {label} | {record.get('status','unknown')} | {detail} |")
 lines+=['','The small apparent order-12 improvement on a coarse curvature-lattice grid was rejected after refinement; see `certs/rbf_lattice_grid_check.json`.']
 write_text(ROOT/'PROGRESS.md','\n'.join(lines)+'\n');return {'gpu_evaluations':totals[0],'refinement_evaluations':totals[1]}
if __name__=='__main__':print(update())
