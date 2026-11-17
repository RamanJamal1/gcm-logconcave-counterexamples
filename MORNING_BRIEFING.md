# Ongoing overnight research — provisional briefing

The scheduled run continues until 2026-11-17 14:02 UTC. This is not a final report.

The exact minimal order remains unresolved. Exact rational sum-of-squares certificates establish the sharp Gaussian lower bounds at orders 6, 7, and 8, under the analytic endpoint hypotheses detailed in `paper/analytic_scope.tex`. Additional exact rational certificates prove nonnegative signs through order 11 (`certs/exact_quotient_m*.json`), independently verified using standard-library rational arithmetic. An explicit strictly log-concave order-17 witness has a rigorous Arb certificate. The resulting current bracket is **12 ≤ m* ≤ 17**.

The simple witness is

`p(x) = exp(-x^2/36) * (1 + cos(x)/20 + cos(2*x)/100000) / Z`,

where `Z = 6*sqrt(pi)*(1 + exp(-9)/20 + exp(-36)/100000)`.

Its curvature satisfies `V'' > 1/400` everywhere, and its signed order-17 entropy derivative is less than `-3/20`. See `certs/rationalized_m17_2.json` and `certs/independent_spatial_m17.json` and `paper/explicit_upper_bound.tex` for the exact parameters, interval and error bounds. The heat generator is one-half the Laplacian, and H is Boltzmann entropy (integral of p log p).

The main A6000 campaign is still running. Live counts are in `checkpoints/campaign_A.json`; the supplementary exact-parameter sweep is in `checkpoints/series_campaign.json`. The earlier complicated rational order-17 witness was independently checked by differentiating the exact heat density with Arb Taylor series before Gauss–Hermite quadrature; its 2048-node numerical value agrees with the rigorous finite-Fourier-series result to about 1e-12 absolute error. That numerical cross-check is not itself a proof.

Hardware, methods, final evaluation totals, bibliography, certificates and final limitations will be consolidated after the scheduled run. A GitHub issue will be posted with the completed briefing then. No exact minimality or novelty claim is being made.
