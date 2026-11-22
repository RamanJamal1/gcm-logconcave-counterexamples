# Research log — 2026-11-21 (A100 session)

Goal: determine m* exactly. The verified bracket is unchanged: **14 ≤ m* ≤ 16**.
By the downward-propagation lemma, an exact certificate S_15 ≥ 0 alone would give m* = 16;
a certified log-concave witness with S_15 < 0 would give m* ≤ 15.

## New results
- **Zou–Fan–Gao–Wang family threshold = 33** (`checkpoints/zou_family_threshold.json`, `src/zou_threshold.py`).
  arXiv 2608.30275 proves f_m fails only for "sufficiently large m". Rigorous Arb series enclosures
  (`certify_fourier_series`) give S_M(f_M) > 0 for 6 ≤ M ≤ 32 and S_M(f_M) < 0 for every 33 ≤ M ≤ 400 evaluated
  (exact members: f_32 > 0, f_36 < 0). The order-16 witness here improves on that family by 17 orders.

## Evidence (not proof)
- **GPU search, ~1.2e9 evaluations** of the Gaussian-coupled periodic reduction
  S_m ≈ ½(m−1)!κ^m + m! Σ_n C(m−1,n−1) κ^(m−n) h_n (`src/periodic_cuda.py`), via CEM over K = 3–8 cosine modes,
  with/without sine modes, random and witness-continuation starts (`src/gpu_massive_search.py`, `checkpoints/ms_*.json`,
  `checkpoints/massive_search_m15.json`, `_m14.json`). Objective Q_m = S_m / (½(m−1)!κ^m), Gaussian = 1.
  Best Q_15 ≈ 1.0068 and Q_14 ≈ 1.0062 (K5 cos+sine); nothing below 1. At the witness scale (a_1 ≈ 1/15, κ ≈ 1/14),
  Q_15 stays ~1e7 while Q_16 ≈ −2.9e6. This supports the conjecture **m* = 16** (Gaussian minimizes Q_15), but the
  surrogate omits exponentially small filter terms and CEM can miss narrow regions.

## Lower-bound attempt (order 14/15 exact SOS)
- Regenerated `native_sos_m14/m15.npz` (m14: blocks 135 + 101, 708 constraints, 28426 variables).
- New FP64 GPU infeasible-start HKM interior point (`src/gpu_ipm_feas.py`): stalls at primal residual ~2e-7 relative;
  dual ‖y‖ grows to 1e7 with bᵀy ≈ 3.1e5 — a ~1e10 dynamic range against the 1e-5 margin, beyond FP64.
- GPU Douglas–Rachford/Anderson from the IPM point: min scaled eigenvalue −3.4e-4 (needs > 0), residual 5e-12.
- sdpap (200-bit) stuck in setup; CLARABEL never finished iteration 0. Diagnostics: `certs/sos_attempt_2026-11-21_m14.json`.
- Interpretation: a precision limit, not infeasibility (order 13 needed double-double SDPA).
  **Next step:** double-double (or 128-bit) version of `gpu_ipm_feas.py` — the Schur system is only 708×708 at m14
  (1039 at m15), so this is cheap; then `preproject_sdp_seed` → `reconstruct_quotient_fast` → `verify_sos_standalone`.

## Infrastructure
- Campaign C searches now keep the GPU at ~90% utilization (best-first certification queue; commit a92d928).
