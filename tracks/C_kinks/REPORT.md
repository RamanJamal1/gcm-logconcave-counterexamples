# Track C: kink interference (interim)

Derivation: p ~ e^{-V}, V convex piecewise linear (knots x_k, slopes s_i, V = s_i y + c_i on segment i).
q = P_1 p = sum_i T_i, T_i = e^{-c_i - s_i x + s_i^2/2} [Phi(x_{i+1}+s_i-x) - Phi(x_i+s_i-x)].
With g_k(x) = e^{-V(x_k)} G(x-x_k): T_i' = -s_i T_i - (g_{i+1} - g_i), so q' = -sum s_i T_i (knot terms telescope).
Taylor jets by recurrences: (k+1) t_{k+1} = -s t_k - (gam^u_k - gam^l_k); Gaussian coefficients (k+1) gam_{k+1} = -(x-x_k) gam_k - gam_{k-1}.
Log-derivatives via k Q_k = sum_j j phi_j Q_{k-j}. Code: kinks.py (log-space, scaled by a common reference).

Validation: u_1..u_16 vs mpmath (120 digits) relative error < 1e-11 (val.py, 3 configs x 3 points).
Scale-free R_m = S_m / int|F_m| q. Trapezoid quadrature, h = 0.05.

Scan: data/map_k2k3.json (R_m, m = 6..16, vs gap d in [0.1, 20], 8 families). All R_m >= 0.
Interference dips near d ~ 2.6 (R_16 down to 2e-4) but no sign change. Plateau for d >= 6 equals the isolated-kink value.
Optimizer logs: logs/opt_n*_m15.log.

## Results (final)
- S_m validation: FP64 sum of F_m*q vs mpmath (110 digits, same grid, 2 configs) agrees to <=3e-12 relative for m=6..16 (val_S.py). m<=13 always >=0 in all scans (no bug flag).
- Extended to m=17..20 using exact polynomials from entropy_sparse.polynomials(20) (ext.py); NOT mp-validated beyond m=16.
- Single kink/isolated knots (d>=6): R_m positive, decays ~x0.3 per m (e.g. a=1: R_14=3.7e-2, R_16=3.5e-3, R_20=1.5e-5). Constant sign as predicted.
- Two/three kinks: interference at gap d~2.6-3.4 lowers R by 10-40x (R_16 -> 2e-4, R_20 -> 9e-7) but never crosses zero.
- DE minimisation (data/opt_*_15_1.json), min R_15: n=2 1.14e-3; n=3 5.5e-4; n=5 3.06e-4; n=8 4.0e-4 (not converged); R_16 n=3 1.5e-4, n=5 2.2e-4 (running); R_14 n=3 1.9e-3. All interior positive optima; no negative value -> no certificate attempted.
- Profile of optimizers over m=6..20: data/profile_opt15.json, all positive, decay ratio slows to ~0.3/m by m=20 (n=5: R_20 = 9e-7).
