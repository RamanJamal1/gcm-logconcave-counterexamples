"""Sparse singleton pivots plus a small exact Schur-complement solve."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import math,numpy as np,scipy.linalg as la
from flint import fmpq,fmpq_mat

def select(keys,cols,layout,scales):
 factors=np.array([1/(scales[bl][i]*scales[bl][j]) for bl,i,j in layout]);A=np.array([[float(d.get(k,0)) for d in cols] for k in keys]);num=A*factors;rn=np.maximum(np.linalg.norm(num,axis=1),1e-100);single={};keyindex={k:i for i,k in enumerate(keys)}
 for j,d in enumerate(cols):
  if len(d)==1:
   row=keyindex[next(iter(d))];strength=abs(num[row,j])
   if row not in single or strength>single[row][0]:single[row]=(strength,j)
 top=sorted(single);chosen=[single[row][1] for row in top];chosen_set=set(chosen);rest=[j for j in range(len(cols)) if j not in chosen_set];bottom=[i for i in range(len(keys)) if i not in single]
 _,_,piv=la.qr((num/rn[:,None])[bottom][:,rest],mode='economic',pivoting=True);tail=[rest[int(j)] for j in piv[:len(bottom)]];pivots=chosen+tail
 return top,bottom,chosen,tail,A,factors

class ExactPivotSolve:
 def __init__(self,keys,cols,top,bottom,chosen,tail):
  self.keys,self.cols,self.top,self.bottom,self.chosen,self.tail=keys,cols,top,bottom,chosen,tail
  self.diagonal=[fmpq(str(cols[j].get(keys[i],0))) for i,j in zip(top,chosen)]
  self.C=fmpq_mat([[fmpq(str(cols[j].get(keys[i],0))) for j in tail] for i in top])
  E=fmpq_mat([[fmpq(str(cols[j].get(keys[i],0))) for j in tail] for i in bottom]);self.inverse=E.solve(fmpq_mat([[int(i==j) for j in range(len(bottom))] for i in range(len(bottom))]),algorithm='dixon')
 def solve_columns(self,top_values,bottom_values):
  bot=self.inverse*fmpq_mat(bottom_values);upper=fmpq_mat(top_values)-self.C*bot
  for i,d in enumerate(self.diagonal):
   for j in range(upper.ncols()):upper[i,j]/=d
  return upper,bot
 def solve_vector(self,rhs):
  upper,lower=self.solve_columns([[rhs[i]] for i in self.top],[[rhs[i]] for i in self.bottom]);return [upper[i,0] for i in range(upper.nrows())]+[lower[i,0] for i in range(lower.nrows())]
