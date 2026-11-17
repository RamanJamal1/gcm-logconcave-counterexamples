# Literature verification and mathematical conventions

Checked 2026-11-17 UTC. Primary HTML was read through the web tool. Direct local HTML/PDF/TeX downloads failed with DNS resolution errors; `/literature/retrieval.json` records all six attempts. The web tool's TeX-source requests also failed (cache miss). Consequently no local full-source archive is claimed.

## Zou, Fan, Gao, Wang (2026)

Primary sources: [abstract](https://arxiv.org/abs/2608.30275), [full HTML v1](https://arxiv.org/html/2608.30275v1). Submitted August 31, 2026. Title and all four authors match the directive.

The convention is **Boltzmann entropy**, not differential entropy:

\[
H(p)=\int p\log p,\quad P_t=e^{t\partial_x^2/2},\quad S_m=(-1)^m\partial_t^m H(P_tp)|_{0}.
\]

Equations (3), (4), (16), (17) give

\[
a_m=2^{-m/4},\quad b_m=2^{-m},\quad \kappa_m=8a_m,\quad
f_m(x)=\frac{e^{-\kappa_mx^2/2}(1+a_m\cos x+b_m\cos2x)}{Z_m},
\]
\[
Z_m=\sqrt{2\pi/\kappa_m}\,[1+a_me^{-1/(2\kappa_m)}+b_me^{-2/\kappa_m}],\qquad
-(\log f_m)''\ge5a_m>0\quad(m\ge5).
\]

Lemma 3 supplies the exact heat evolution, with \(d=1+\kappa t\):

\[
P_t[e^{-\kappa x^2/2}e^{inx}]
=d^{-1/2}e^{-\kappa x^2/(2d)}e^{-n^2t/(2d)}e^{inx/d}.
\]

Equation (19) defines \(\tau=t/d\) and **\(\lambda=\kappa d\)**. The latter is essential when changing variables \(y=x/d\).

Theorem 1 proves

\[
S_m(f_m)=-\frac{11}{96}\left(\frac{3}{2\sqrt2}\right)^m+o\!\left(\left(\frac{3}{2\sqrt2}\right)^m\right).
\]

This is an asymptotic analytic result. It supplies neither a numerical threshold nor certificates at orders 6, 7, 8. There are no finite-order certification parameters to extract from this paper.

## Gu and Sellke (2026)

Primary sources: [abstract](https://arxiv.org/abs/2605.11656), [full HTML v1](https://arxiv.org/html/2605.11656v1). Submitted May 12, 2026. Authors: Yuzhou Gu and Mark Sellke.

They use \(P_t=e^{t\partial_x^2}\) and the same Boltzmann sign convention. Their unrestricted counterexample is

\[
\mu=.002\delta_0+\sum_{j=0}^7c_j(\delta_{1.8+j}+\delta_{-(1.8+j)}),\quad
c=(.034,.093,.134,.123,.075,.031,.008,.001).
\]

Proposition 1 certifies \(0.36<H_\mu^{(5)}(1/3)<0.37\). In the half-Laplacian convention this becomes
\(0.36/32<H_\mu^{(5)}(2/3)<0.37/32\).

Appendix A uses exact rational inputs and 256-bit complex balls. Twenty-two finite integration intervals partition \([0,25]\); symmetry doubles the integrals. Each reported interval contribution has radius \(10^{-4}\). The analytic tail bound is less than \(10^{-20}\).

Theorem 2 establishes existence of a log-concave counterexample through smoothing and backward propagation, without specifying its derivative order. The text explicitly says that this order is at least six. The displayed 17-atom witness does **not** establish an order-five log-concave counterexample.

## Lower-order theorem and later search

[Guillaume Wang, arXiv:2408.13957](https://arxiv.org/abs/2408.13957), *A higher-order Otto calculus approach to the Gaussian completely monotone conjecture*, proves the signs through order five for log-concave inputs in every dimension. This provides the required lower-order benchmark.

Searches for the exact 2608.30275 identifier, its title, and combinations of GCM/log-concavity with sixth order, seventh order, minimal order and late 2026 found indexes and summaries, but no verified later primary paper resolving the minimal log-concave order. This is a bounded literature search, not proof of absence of such a paper.

## Independent implications for the proposed experiment

The following observations are derived here, not attributed to an unverified reference.

1. A certified order-six witness would settle the requested minimal order using the lower-order theorem. An order-seven or order-eight witness supplies only an upper bound unless all smaller eligible orders are proved universally nonnegative. Finite unsuccessful searches cannot provide such a proof.
2. There is no scale normalization in the directive. Set \(p_a(x)=a^{-1}p(x/a)\). Heat scaling gives \(H(P_tp_a)=H(P_{t/a^2}p)-\log a\), hence \(S_m(p_a)=a^{-2m}S_m(p)\). Broadening a Gaussian approaches zero from above; a negative witness, if found, can be dilated towards zero or contracted towards minus infinity. Therefore a strictly positive universal numerical lower bound over the stated class is impossible. For search comparisons use \(S_m(p)\operatorname{Var}(p)^m\), or fix variance.
3. A Gaussian of variance \(v\) must give \(S_m=(m-1)!/(2v^m)>0\) under the half-Laplacian convention. A negative Gaussian control indicates a sign error.
4. High-precision floating-point quadrature and gridwise curvature checks do not by themselves certify the all-real-line statement. Outward-rounded integration enclosures, analytic tails, and a global curvature argument are separate requirements.

## Reproducible queries

- `"2608.30275"`
- `"An Explicit Family of Log-Concave Counterexamples"`
- `"Gaussian completely monotone" "minimal" "log-concave"`
- `"Gaussian completely monotone" "2026" "order six"`
- `"Gaussian completely monotone" "2026" "log-concave" "seventh"`
- `"Gaussian completely monotone" "m=6"`
- `site:arxiv.org "A higher-order Otto calculus approach"`

