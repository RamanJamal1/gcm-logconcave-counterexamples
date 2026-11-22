"""Check (or regenerate) the SHA-256 manifest of every certificate cited in the paper.

Usage, from the repository root:
  python src/check_hashes.py           # check certs/exact_certificate_sha256.json
  python src/check_hashes.py --write   # regenerate it from the files listed below

Hashes are of the exact committed bytes. On Windows, clone with
`git -c core.autocrlf=false clone ...` (or rely on .gitattributes) so that checkout
does not rewrite line endings.
"""
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
MANIFEST=ROOT/'certs'/'exact_certificate_sha256.json'
CITED=(
 [f'certs/exact_sos_m{m}.json' for m in (5,6,7,8)]
 +[f'certs/exact_quotient_m{m}.json' for m in (9,10,11,13)]
 +[f'certs/exact_mckean_m{m}.json' for m in (6,7,8)]
 +[f'certs/exact_mckean_quotient_m{m}.json' for m in (9,10,11)]
 +['certs/paired15_m16.json','certs/independent_paired15_m16.json','certs/exact_min_curvature_paired15_m16.json',
   'certs/witness_order_table.json']
 +[f'certs/witness_order_table_paired15/m{m}.json' for m in range(6,21)]
 +['checkpoints/zou_family_threshold.json']
)

def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def main():
 if '--write' in sys.argv[1:]:
  MANIFEST.write_text(json.dumps({p:digest(p) for p in CITED},indent=2)+'\n',newline='\n')
  print(f'wrote {len(CITED)} hashes to {MANIFEST.relative_to(ROOT).as_posix()}');return 0
 stored=json.loads(MANIFEST.read_text());bad=0
 for path,expected in stored.items():
  actual=digest(path) if (ROOT/path).exists() else None
  ok=actual==expected;bad+=not ok
  print(('OK      ' if ok else ('MISSING ' if actual is None else 'MISMATCH')),path)
 print(f'{len(stored)-bad}/{len(stored)} hashes match')
 return 1 if bad else 0

if __name__=='__main__':
 sys.exit(main())
