# Track E — GPU double-double feasibility solver (work in progress, recovered)

Recovered from the engineer's local copies after the A100 instance (qsklosxz) was lost on 2026-11-21.
Nothing here is a certificate. Status as reported by the engineer:

- `dd.py`: double-double arithmetic in torch/CUDA + Ozaki-style DD matmul. Tested vs mpmath: ~1e-32 (ops), ~1e-33 (matmul).
- `ddla.py`: blocked DD Cholesky / triangular solves. cond-1e20 test (n=50): forward error 6e-15 → 5e-29 after one refinement.
- `build_dd.py`: exact rationals (setup pkl from tracks/D_order14/prep_sparse.py) → DD constraint data. Built m13 (478 rows, 13 s), m14 (708 rows, 28 s).
- `solve.py`: dual-barrier Newton, min −bᵀy − logdet(tI − mat(Aᵀy)), X = Z⁻¹; fp64 warm-up then DD phase (Newton–Schulz DD inverse, DD Schur + DD Cholesky, DD refinement); writes 40-digit seed JSON. Final DD projection written but untested.
- Order 13 (validation): converged (~105 of 150 iterations, ~11 s/iter, ~1640 s total); gradient floor ~1e-21 (Newton–Schulz accuracy);
  min eig of X: 4.9e-6 (block 0), 1.6e-4 (block 1). Seed/reconstruct/verify NOT yet run.
- Findings: (1) with margin δ = 1e-5 the solver found essentially no interior at order 13 (the known certificate's min scaled
  eigenvalue is 1.0004e-5), so δ = 0 was used — the 1e-5 margin used elsewhere may be too strict; (2) FP64 Newton steps are
  wrong (bad Newton decrement) because the Schur matrix is too ill-conditioned; only DD Cholesky fixed it.
- Not done: order-13 reconstruct + strict verify; orders 14, 15.
