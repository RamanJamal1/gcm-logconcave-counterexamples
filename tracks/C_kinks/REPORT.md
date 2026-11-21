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
