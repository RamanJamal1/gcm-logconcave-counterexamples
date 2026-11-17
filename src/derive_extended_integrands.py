"""Export exact sparse entropy polynomials through order seventeen."""
from entropy_sparse import polynomials
from pathlib import Path
polys=polynomials(17);lines=['# Generated from the exact sparse entropy recurrence.']
for m,p in polys.items():
 terms=[]
 for e,c in p.items():
  factors=[f'u[{j+1}]**{power}' for j,power in enumerate(e) if power]
  terms.append('('+str(c.numerator)+'/'+str(c.denominator)+')*'+'*'.join(factors))
 lines += [f'def integrand_{m}(u):','    return sum(('+','.join(terms)+',))','']
lines.append('INTEGRANDS={'+','.join(f'{m}:integrand_{m}' for m in polys)+'}')
Path('src/generated_extended_integrands.py').write_text('\n'.join(lines)+'\n')
