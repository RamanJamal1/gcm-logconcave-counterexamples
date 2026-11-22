# Track X1 — positive-time compactification (Tweedie variables)

Verdict: **Tasks 1–4 VERIFIED. Task 5 is NEGATIVE.** I prove below (Thm 5) that the enlarged ansatz cannot certify any
order the existing ansatz cannot certify. Script: `constants.py` (python-flint Arb at 256 bits, plus sympy checks;
every check raises explicitly; it passes under `python3 -O`). Output: `constants.json`.
Context: PROGRESS.md records an Arb-certified order-16 witness with S_16 < 0, so no ansatz of any kind can certify m = 16.

## Setup and the existing form (from src/)
The project's setup: H(p)=∫p log p, ∂_t p = ½∂²p, u_j = (log p)^{(j)}, S_m(p) = E_p F_m(u_2..u_m) (src/derive_integrands.py,
src/entropy_sparse.py). Give u_j weight j. Then F_m is weight-homogeneous of weight 2m.
The existing certificate (src/sos_quotient.py `setup`, also used by tracks/D_order14/prep_sparse.py) is
F_m ≡ z_mᵀQ_0 z_m + (−u_2)·z_{m−1}ᵀQ_1 z_{m−1}, taken mod the integration-by-parts relations E[u_1·M] = −E[∂_x M]
(the `reduce`, which eliminates u_1). Here z_w is the vector of monomials of weight w, Q_i ⪰ 0, and −u_2 ≥ 0 is the log-concavity multiplier.
All of this is weight-graded, and IBP preserves weight.

## Thm 1 (Tweedie identities, exact)
Let p be a probability density, s>0, q = p * N(0,s). Let μ_y be the posterior, dμ_y(x) ∝ p(x)e^{−(x−y)²/(2s)}dx.
Then for every y:
(log q)'(y) = (E[X|Y=y] − y)/s,  (log q)''(y) = −1/s + κ_2(X|Y=y)/s²,  (log q)^{(k)}(y) = κ_k(X|Y=y)/s^k for k ≥ 3
(this holds for all k, in particular k ≤ 16).
*Proof.* q(y) = (2πs)^{−1/2} e^{−y²/(2s)} ∫ p(x)e^{−x²/(2s)} e^{xy/s} dx. So log q(y) = c − y²/(2s) + K(y/s), where K is the
cumulant generating function of the finite measure ν = p(x)e^{−x²/(2s)}dx. K is finite and analytic on all of R because ν has Gaussian tails.
The exponentially tilted law ν_θ ∝ e^{θx}ν at θ = y/s is μ_y, and K^{(k)}(θ) = κ_k(ν_θ). The chain rule gives a factor s^{−k}. ∎
(The script checks this numerically to 1e−20 for p ∝ x²e^{−x}1_{x>0}, s=1, k=2,3,4.)

**Normalization.** u_k := s^{k/2}(log q)^{(k)}. Then u_2 = −1 + Var(μ_y)/s and u_k = κ_k(μ_y)/s^{k/2} for k ≥ 3.
In other words, u_k = κ_k of the posterior rescaled by s^{−1/2}, which is a 1-strongly log-concave law (see below).

## Thm 2. −1 < u_2 ≤ 0 if p is log-concave
μ_y has density e^{−V}, with V = −log p + (x−y)²/(2s) convex and V'' ≥ 1/s in the distributional sense, i.e. μ_y is 1/s-strongly log-concave.
Brascamp–Lieb gives Var(μ_y) ≤ E[1/V''] ≤ s, so u_2 ≤ 0. If p is not smooth, approximate V by smooth V_ε with V_ε'' ≥ 1/s. Alternatively, Caffarelli's
contraction theorem (Thm 3) gives Var(T(Z)) ≤ E|T(Z)−T(0)|²... ≤ s directly.
A non-degenerate law has Var > 0, so u_2 > −1. Both bounds are sharp: the limit p → δ gives u_2 → −1, and the limit p wide/flat gives u_2 → 0. ∎
Consequence: the multipliers −u_2, 1+u_2 and (1+u_2)(−u_2) are all ≥ 0 on Q_s := {P_s p : p log-concave}.

## Thm 3. Rigorous constants
Let X be 1-strongly log-concave on R. Then |κ_k(X)| ≤ C_k, where
C_k := Σ_{π ∈ Π(k), no singleton blocks} (|π|−1)! ∏_{B∈π} M_{|B|},  with M_2 = 1 and M_j = 2^j Γ((j+1)/2)/√π for j ≥ 3.
Since s^{−1/2}X_post is 1-strongly log-concave, the bound for the Tweedie variables is **|u_k| ≤ C_k for all y and s.**
*Proof.* Write μ_j = E(X−EX)^j.
(a) Moment–cumulant formula: κ_k = Σ_{π, no singletons} (−1)^{|π|−1}(|π|−1)! ∏ μ_{|B|}, because κ_k is shift-invariant for k ≥ 2.
The script self-checks this exactly on Exp(1), where κ_k = (k−1)!, for k = 3..16.
(b) μ_2 ≤ 1 by Brascamp–Lieb.
(c) Caffarelli's contraction theorem (in 1D this is elementary: T = F_X^{−1}∘Φ has T' = φ(Z)/f_X(T(Z)) ≤ 1) gives X = T(Z) with T 1-Lipschitz.
Let X' = T(Z') be an independent copy. Jensen in X' gives E|X−EX|^j ≤ E|X−X'|^j. Then E|X−X'|^j ≤ E|Z−Z'|^j = 2^{j/2}E|Z|^j = M_j.
(d) The triangle inequality over the partition sum finishes the proof. ∎
Arb values (256-bit balls; radii are far below the digits shown; full balls are in constants.json):

| k | C_k (proven) | k | C_k (proven) |
|---|---|---|---|
| 3 | 4.51351666838 | 10 | 4279669.91254 |
| 4 | 15 (exact) | 11 | 5.47950095788e7 |
| 5 | 81.2433000309 | 12 | 7.71522762166e8 |
| 6 | 533.718327158 | 13 | 1.18485544207e10 |
| 7 | 4035.08390153 | 14 | 1.97152893631e11 |
| 8 | 36284.8073775 | 15 | 3.53275191875e12 |
| 9 | 372023.119378 | 16 | 6.78236431432e13 |

For even k, C_k is rational: odd M_j never appear alone, but products of two odd-index M_j bring in 1/π. The Arb balls are what is certified.
Sharper constants: these bounds are crude. For example, the half-Gaussian gives |κ_3| ≈ 0.22, against C_3 ≈ 4.51. Using (π/2)^j E|Z|^j (Maurey–Pisier) instead would be *worse*.
**No sharper constant is claimed as proven.** By Thm 5, sharpening them cannot help anyway.

## Thm 4. Equivalence across t → 0
(i) Semigroup: S_m(P_s p) = (−1)^m h^{(m)}(s), with h(t) = H(P_t p).
(ii) Scaling: put p_λ(x) = λp(λx). Then H(p_λ) = H(p) + log λ and P_t(p_λ) = (P_{λ²t}p)_λ, so **S_m(p_λ) = λ^{2m} S_m(p)**.
Also (P_s p)_{s^{−1/2}} = P_1(p_{s^{−1/2}}), so Q_s is a rescaling of Q_1 and the sign question on ∪_s Q_s reduces to Q_1.
(iii) Q_s ⊂ regular log-concave densities (Prékopa). For q ∈ Q_s:
- u_k is bounded for k ≥ 2 (Thm 3);
- |u_1| grows at most linearly;
- q decays at least exponentially.
So all IBP boundary terms used by derive_integrands vanish, and S_m(q) = E_q F_m is rigorous on Q_s. This is *cleaner* than at t = 0.
(iv) Hence "S_m ≥ 0 on Q_1" ⇔ "h_p is (−1)^m-monotone at order m on (0,∞) for every log-concave p".
This implies S_m(p) ≥ 0 for every log-concave p for which h^{(m)}(s) → h^{(m)}(0+) as s↓0.
That covers the project's class (Gaussian tails, bounded smooth perturbations), by dominated convergence on E_{P_s p}F_m(u).
The converse direction is trivial, since Q_1 ⊂ LC.
**Gap (minor, definitional):** for general log-concave p (e.g. uniform), S_m(p) at t = 0 is only defined as the limit s↓0, and there the
positive-time statement is the natural definition. There is no gap in the order of limits for the regular class.

## Thm 5 (negative: the enlarged ansatz gives no new certificates)
Suppose that, mod IBP and at s = 1,
F_m ≡ Σ_i σ_i g_i, with σ_i SOS and g_i products of {1, −u_2, 1+u_2, C_k²−u_k², …}, or more generally any polynomials g_i ≥ 0 on Q_1.
Then the lowest-weight part of this identity is already a certificate of the existing type. Precisely, it is F_m ≡ Σ σ_i^{low} g_i^{low} mod IBP, where
- the σ_i^{low} are SOS of no larger Gram size;
- each g_i^{low} is a positive constant times a power of (−u_2), or more generally a weight-homogeneous polynomial that is ≥ 0 on all regular LC densities.
*Proof.* Split everything by weight. IBP and F_m respect the grading, so each weight component of the identity holds separately.
Let w_0 be the minimal weight of the nonzero lowest components σ_i^{low}g_i^{low}, and suppose w_0 < 2m. Then Σ_{weight w_0} σ_i^{low}g_i^{low} ≡ 0 mod IBP.
Every g^{low} is ≥ 0 on regular LC densities: apply g ≥ 0 on Q_1 to P_1(p_λ), rescale back, and let λ → 0. Rescaling multiplies a weight-w part by λ^{−w}, so the lowest component dominates.
Hence each E_p[σ_i^{low}g_i^{low}] = 0 for all regular LC p. The jets (u_2,…,u_r)(0) of Gaussian-perturbed densities fill an open set with u_2 < 0, so σ_i^{low}g_i^{low} ≡ 0. That contradicts the minimality of w_0.
So w_0 = 2m, and the weight-2m component is the stated certificate. ∎
The box multipliers have lowest part C_k² (a constant), and 1+u_2 has lowest part 1. So they collapse to plain SOS.
**Compactification therefore adds nothing for weight-homogeneous targets.** It could only help through genuinely new *homogeneous* valid constraints (e.g. other
LC-valid inequalities on jets), which are independent of the compactification.
It could also help for non-homogeneous targets, e.g. combinations Σ_m c_m s^m S_m(q) arising from Taylor expansion in s. Those are a different problem and are not
equivalent to the single-order sign question.

## What was checked
- `python3 -O constants.py`: moment–cumulant self-test (exact sympy, k = 3..16); Tweedie identity (numerical, 1e−20); Arb C_k.
- Thms 2–5 are pen-and-paper proofs relying on standard results (Brascamp–Lieb, Caffarelli 2000, Prékopa). No SOS certificate was produced.
