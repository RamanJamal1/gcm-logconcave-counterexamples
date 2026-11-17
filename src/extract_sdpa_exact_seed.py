"""Preserve the SDP decimal output as exact rationals before affine repair."""
import argparse,json,math
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1]
def extract(m):
 state=json.loads((ROOT/f'checkpoints/sdpa_segments_m{m}.json').read_text());r=json.loads((ROOT/f'certs/sos_segment_m{m}.json').read_text());scales=r['basis_scales'];dims=list(map(len,scales));grams=[[['0' for _ in range(n)] for _ in range(n)] for n in dims]
 with open(ROOT/state['initial_file']) as f:
  next(f)
  for line in f:
   k,block,i,j,value=line.split();k,block,i,j=map(int,[k,block,i,j])
   if k!=2:continue
   block-=1;i-=1;j-=1;v=Fraction(value)+(Fraction(1,100000) if i==j else 0);v*=Fraction(math.factorial(m-1),scales[block][i]*scales[block][j]);grams[block][i][j]=grams[block][j][i]=str(v)
 r['gram_matrices']=grams;r['source_checkpoint']=state['initial_file'];r['exact_decimal_seed_only']=True
 path=ROOT/f'certs/sos_segment_m{m}_exact_seed.json';path.write_text(json.dumps(r,indent=2)+'\n');print(path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('order',type=int);a=p.parse_args();extract(a.order)
