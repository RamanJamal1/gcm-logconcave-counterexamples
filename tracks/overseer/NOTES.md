## 2026-11-21 22:15Z overseer note (M15 dual-barrier solve, PID 9106)
- m15 |g|inf trajectory (every 8 dd its, from it33): 290,262,237,216,199,185,173,163,154,146,138,132,126. lam2 flat ~3.05, alpha=1, Xmax falling.
  m13 (P1, same solver, delta 0) showed the SAME lam2~3.1 plateau, then converged (it101 lam2=0.10, |g|=0.10).
  Difference: m13 |g| fell ~linearly (-0.6/it) and accelerated; m15 decrements are shrinking (28,25,21,17,14,12,10,9,8,8,6,6).
  => Not yet decidable. Two readings: (a) slow approach to an analytic center (margin>0, ~500+ more its = 1.5-2 h);
  (b) stall at |g| ~ O(50-90) => near-boundary / margin<=0.
- --maxit 300 will likely end before convergence. Do NOT read "not converged at 300" as margin<0. Resume from checkpoint (--resume).
- Decisive test instead of waiting: take the accumulated direction d = y_k - y_{k-50} (or dy). If A^T d <= 0 (PSD-negative)
  and b^T d > 0 hold, round d to rationals and certify exactly => dual ray => this SOS form is infeasible at m=15.
  If A^T d has a clearly positive eigenvalue, the iterates are not running to infinity; keep solving.
- FYI: the killed P1 m13 solve was converging (it101 lam2=0.103) when the overseer stopped it per the user's skip directive.
