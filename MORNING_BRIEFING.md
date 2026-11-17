# Ongoing overnight research — provisional briefing

The scheduled run continues until 2026-11-17 14:02 UTC. This is not a final report.

The exact minimal order remains unresolved. Exact rational sum-of-squares certificates establish the sharp Gaussian lower bounds at orders 6, 7, and 8, under the analytic endpoint hypotheses detailed in `paper/analytic_scope.tex`. Additional exact rational certificates prove nonnegative signs through order 10 (`certs/exact_quotient_m*.json`), independently verified using standard-library rational arithmetic. An explicit strictly log-concave order-17 witness has a rigorous Arb certificate. The resulting current bracket is **11 ≤ m* ≤ 17**.

The simple witness is

`p(x) = exp(-33*x^2/2000) * (1 + cos(x)/32 + cos(2*x)/270000) / Z`,

where `Z = sqrt(2000*pi/33)*(1 + exp(-500/33)/32 + exp(-2000/33)/270000)`.

Its curvature satisfies `V'' > 1/2000` everywhere, and its signed order-17 entropy derivative is less than `-1/20`. See `certs/simple_fourier_m17.json` and `paper/explicit_upper_bound.tex` for the exact parameters, interval and error bounds. The heat generator is one-half the Laplacian, and H is Boltzmann entropy (integral of p log p).

The main A6000 campaign is still running. Live counts are in `checkpoints/campaign_A.json`; the supplementary exact-parameter sweep is in `checkpoints/series_campaign.json`. The earlier complicated rational order-17 witness was independently checked by differentiating the exact heat density with Arb Taylor series before Gauss–Hermite quadrature; its 2048-node numerical value agrees with the rigorous finite-Fourier-series result to about 1e-12 absolute error. That numerical cross-check is not itself a proof.

Hardware, methods, final evaluation totals, bibliography, certificates and final limitations will be consolidated after the scheduled run. A GitHub issue will be posted with the completed briefing then. No exact minimality or novelty claim is being made.
