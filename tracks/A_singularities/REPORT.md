# Track A — complex-time singularities (numerical, periodic surrogate; NOT a proof)

Written by the lead from the Track A engineer's report (commits d462f70, 7d58c22). Code and data in this directory
(`data/darboux_*.txt` m=6..399, `sing_*.json`, `spectrum_*.json`, `kappa_scan.json`, `optrho_*.json`).

## Conventions
σ = −τ, T = −t. g(σ) = ⟨R log R⟩, R = 1 + Σ aₙ e^{n²σ/2} cos nx; Pₙ = n!·gₙ; S_m = m!·[T^m] F̃(T),
F̃ = −½ log(1−κT) + g(T/(1−κT)). Surrogate omits Gaussian-filter terms; it reproduces the certified witness to 1–10%
(S₁₄..S₂₀ = 2.021, 3.281, −0.848, −36.64, −34.12, 3682, 75233 vs certified 2.023, 3.295, −0.770, −36.17, −31.18, 3700.7, 75350).

## Results
- **Singularities of g** are contour pinches (double zeros R = R_x = 0, x complex in general), local form
  (β/3)(2α(τ−τ*)/β)^{3/2} with α = R_τ, β = R_xx — exponent 3/2, not logarithmic. Only the Gaussian singularity
  T = 1/κ is logarithmic; g-singularities form lattices accumulating there.
- **Darboux:** S_m ~ Γ(m−3/2)/Γ(−3/2) Σ_j A'_j T_j^{−m} + ½(m−1)!κ^m, A'_j = A_j(1+κσ_j)^{3/2}.
  Fitted amplitudes match predictions to 1.00 ± 0.01. Relative error O(1/m) (≈15–40/m); **asymptotic regime starts at
  m ≳ 25–40; order 15 is pre-asymptotic for every configuration.** Nearest-singularity-only fails until m ≈ 150
  (3–4 singularities within 3% in |T|).
- Nearest singularities: witness — real σ = 3.6538 (T = 2.8976) with a complex pair at |T| = 2.9164 ⇒ S_m > 0
  asymptotically; 2-mode — complex pair |T| = 3.756 ⇒ infinitely many sign changes (S < 0 at m = 60, 100, 300);
  Zou f₃₂ — complex pair |T| = 8.086.
- **Spectral-line form (exact dual):** g = Σ_d C_d e^{dσ/2}, d = Σ nᵢ², and
  S_m = Gauss + Σ_d C_d·Kern_m(d/2; κ) with Kern_m(ℓ) = m!·[T^m] e^{ℓσ(T)} > 0.
  Witness lines: d=2: +a₁²/4; d=6: −a₁²a₂/8; d=12: ≈ a₁³a₃/24 (<0); d=18: +a₃²/4. Dominant line: d=2 for m ≤ 14,
  d=6 at m=16, d=12 at m=18, d=18 from m=20. At m=16: positive mass 66.45 vs negative −67.30.
  Negativity = window where dominance hands over from the positive λ=1 line to a negative line, before a positive
  higher line takes over.
- **Curvature-margin metric** ρ_m = κ*_m/κ_min (κ_min = log-concavity threshold, κ*_m = largest κ with S_m < 0);
  S_m < 0 achievable iff ρ_m ≥ 1. Hill-climbs over 3–7 cosine modes (two starts, six runs) converge to
  **ρ₁₅ = 0.846, ρ₁₆ = 1.091, ρ₁₇ = 1.423** (≈1.29× per order). Random screens (~450k) never gave a first negative
  order below 17.

## Verdict and limitations
S₁₅ < 0 in Gaussian×cosine families looks implausible (≈15% κ-margin short) but is not excluded; no rigorous
obstruction derived. ρ₁₅ > 1 would need the negative hand-off before order 12 without inflating κ_min — suggests
non-lattice frequency ratios, sine modes, or non-Gaussian envelopes. Limitations: surrogate only; empirical
principal-sheet rule (|Im x*| < 0.1); incomplete singularity search in dense clusters; leading-order Darboux only;
local FP64 optimisation; no certification.
