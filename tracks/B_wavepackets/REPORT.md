# Track B: localized wave packets on convex bases (no sign change at order 15 or 14)

Family: log p = -V + log(1 + sum_j phi_j(x) sum_k a_jk cos(k w_j x + th_jk)); V = eps x^2/2 + {A sqrt(x^2+L^2) | A log cosh(x/L) | gam x^4 | none}, optional tilt; 1-2 (mirrored) Gaussian-window packets; K=3..5.
Evaluator wp.py: exact Taylor-jet recurrences, u_2..u_16 in FP64 torch, ~1000 evals/s (B=256, 2049 pts, A100).
Validation: Gaussian exact to 1e-12; witness S_15=3.29476, S_16=-0.76971; val_mp.py (python-flint arb series, 200 bit, exact sparse integrand) reproduces witness and matches FP64 on same grid ~1e-9 rel for all finalists (data/final_*.json, with grid/domain convergence rows).
Search: CMA-ES, objective asinh(rho_15/1e-10), rho_m = S_m / int (G_m)^+ p, G_m=(-1)^m F_m. Constraints: margin -max(u2)/E_p[-u2] >= 1e-3 (witness 1.5e-3), min r >= 0.5. a_k = z_k (k w)^-15 parametrization. data/old_param = first-generation (bad scaling, R objective; ignore).

Baselines (data/baselines.json): Gaussian R=rho=Shat=1. Witness: R15=2.71e-8, rho15=5.42e-8, Shat15=1.16e7; R16=-1.74e-10, rho16=-3.49e-10. Gauss x Fourier5: R15=0.0251, rho15=0.0489.

Best rho15 (all positive): periodic 1-freq K3/K5 4.74e-8/4.37e-8; sqrt sym 4.68e-8; sqrt/logcosh asym+tilt 1.5e-7/1.6e-7; logcosh sym 4.62e-8; quartic 4.65e-8; fixed w=20 5.5e-8; logcosh 2 packets K3/K4 1.6e-8/1.28e-8 (S16=-739/-8256); global 2 frequencies 1.38e-8.
No S_15<0 or S_14<0; no certification attempted.

Mechanism: a packet needs V'' >= ~a w^2 on its support, so p decays within ~1/(w sqrt(a)); oscillations per std are bounded as in the global family. Optimizer sends A->0, window->infinity. Sharp bases inflate u_15 (A 15!/L^14) and hence the positive u_15^2/2 term. Second variation of S_m (m=14..16) about a Gaussian under cos(wx) is positive for w in 0.5..6. Only gain (3-4x lower rho15 than witness) is from a second incommensurate frequency, which works globally too, not from locality.
