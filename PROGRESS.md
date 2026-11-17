# GCM research progress

Updated: 2026-11-17T11:48:40.830067+00:00

Verified bracket: 14 ≤ m* ≤ 16. Exact minimality remains unresolved.

| Campaign | Saved status | FP32 candidates | FP64 refined candidates |
|---|---|---:|---:|
| campaign_A | bounded_run_complete | 11,468,800 | 154,112 |
| campaign_B | bounded_run_complete | 70,057,984 | 547,328 |
| campaign_C | running | 18,284,544 | 142,848 |
| higher_campaign | stopped_for_new_order12_family | 1,630,208 | 25,472 |
| polynomial_campaign | interrupted_exit143 | 32,768 | 512 |
| polynomial_wide_campaign | bounded_run_complete | 884,736 | 13,824 |
| harmonic_refinement_campaign | bounded_run_complete | 0 | 1,176 |

Recorded production totals: 102,359,040 FP32 candidate evaluations; 885,272 FP64 refinements. Warmups and readiness runs are excluded. Interrupted partial generations may not be counted.

Sharp proven normalized bounds: S6 Var^6 ≥ 60; S7 Var^7 ≥ 360; S8 Var^8 ≥ 2520.
Nonnegative signs cover orders 6–13, including the recorded heat-flow propagation steps. Sharp Gaussian-bound certificates cover orders 6–11.
An order-16 witness has a spatial Arb certificate; the earlier order-17 witness has independent Fourier-series and spatial certificates.
See MORNING_BRIEFING.md, checkpoints/research_status.json, and paper/draft_note.tex.

## Additional work in progress

Numerical search values and SDP iterates in this table are not proofs.

| Attempt | Status | Recorded work |
|---|---|---|
| Order-12 high-precision SDP | stopped_after_verified_order13_certificate | segment 14; affine residual 9.61e-05 |
| Order-13 high-precision SDP | stopped_after_verified_order13_certificate | segment 4; affine residual 141 |
| Order-15 high-precision SDP | stopped_for_exact_row_preconditioning | segment 1; affine residual 2.07e+03 |
| Order-15 two base frequencies | bounded_run_complete | 12,348 CPU objective evaluations |
| Order-15 positive curvature bumps | bounded_run_complete | 4,400 CPU objective evaluations |
| Order-12 curvature lattice | bounded_run_complete | 30,384 CPU objective evaluations |

The small apparent order-12 improvement on a coarse curvature-lattice grid was rejected after refinement; see `certs/rbf_lattice_grid_check.json`.
