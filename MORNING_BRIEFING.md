# Overnight research results: log-concave GCM

**Provisional report — research is still running.**
Updated 2026-11-17 11:15:30 UTC. Scheduled interval: 2026-11-17T06:02:55.131528+00:00 to 2026-11-17T14:02:00+00:00.

**The exact minimal order remains unresolved. The verified bracket is 12 ≤ m* ≤ 16.**

Exact rational certificates establish nonnegative signs at orders 6, 7, 8, 9, 10, 11. The positive-time smoothing and endpoint hypotheses are stated explicitly in the note. A strictly log-concave order-16 density has a rigorously negative entropy derivative.

The convention is Boltzmann entropy `H = integral p log p`, with heat generator `(1/2) d²/dx²`. The opposite entropy convention reverses the signs.

## Concrete counterexample

```text
p(x) = exp(-x²/30)
       * (1 + cos(x)/16 + cos(2*x)/48000 - cos(3*x)/20000000) / Z
Z = sqrt(30*pi)
    * (1 + exp(-15/2)/16 + exp(-30)/48000
         - exp(-135/2)/20000000)
```

Exact global minimum curvature: `min V'' = 76658/843768795 > 1/12000`.
Certified signed derivative: `S16 < -3/4`, with an enclosure near `-0.76099654207878681651`.

[Primary interval certificate](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/simple_kappa_48000_m16.json).
[Exact global minimum-curvature proof](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/exact_min_curvature_simple_48000_m16.json).
[Independent interval calculation](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/independent_simple_kappa_48000_m16.json). The spatial score-polynomial and exact heat-density calculations agree. Both include rigorous infinite-tail bounds.

## Orders 6, 7, and 8

| Order | Sharp proven lower bound for S_m Var(p)^m | Best recorded numerical value |
|---|---:|---:|
| 6 | 60 | 60.000000000019 |
| 7 | 360 | 360.00000000001 |
| 8 | 2520 | 2520 |

The sharp bound `S_m Var(p)^m ≥ (m-1)!/2` is independently verified throughout orders 6–11; the additional constants at orders 9, 10, and 11 are 20,160, 181,440, and 1,814,400.

An additional exact certificate excludes the entire family `p ∝ exp(beta*x) cosh(x)^(-alpha)`, `alpha > |beta|`, at orders 2–17. Its beta-moment and Bernstein identities were independently checked; this is a family-specific result, not a larger universal lower bound.

The Gaussian uniquely attains these proven bounds. For a normalized minimizing sequence, the relative score discrepancy from the matching Gaussian tends to zero; the note gives an explicit bound. Numerical values can differ in their last digits because of rounding; they are not the proof. The infimum of the unnormalized derivative is zero under dilation, so normalization is essential.

[certs/exact_mckean_m6.json](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/exact_mckean_m6.json), [certs/exact_mckean_m7.json](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/exact_mckean_m7.json), [certs/exact_mckean_m8.json](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/exact_mckean_m8.json).

## Hardware and actual work

GPU: NVIDIA RTX A6000, 49,140 MiB reported memory. The environment exposed 8 CPU cores and 62.8 GiB host RAM; these observed values differ from the requested 6-core/48-GB specification. The filesystem reported about 206.9 GB total capacity. PyTorch/CUDA executed the search; it was not a CPU-only run.

Production FP32 candidate evaluations: **90,562,560**.
Recorded FP64 refined candidates: **793,088**. These are evaluations, not necessarily unique densities; warmups/readiness tests and uncheckpointed interrupted partial generations are excluded.

| Campaign | Saved status | FP32 evaluations | FP64 refinements |
|---|---|---:|---:|
| campaign_A | bounded_run_complete | 11,468,800 | 154,112 |
| campaign_B | bounded_run_complete | 70,057,984 | 547,328 |
| campaign_C | running | 6,488,064 | 50,688 |
| higher_campaign | stopped_for_new_order12_family | 1,630,208 | 25,472 |
| polynomial_campaign | interrupted_exit143 | 32,768 | 512 |
| polynomial_wide_campaign | bounded_run_complete | 884,736 | 13,824 |
| harmonic_refinement_campaign | bounded_run_complete | 0 | 1,152 |

Campaign B used 65,536 simultaneous FP32 candidates, selected 512 elites, and used FP64 refinement with 2,049- and 4,097-point resolution checks. Campaign C uses a periodic GPU approximation only to screen rationalized parameters; retained negative results require full spatial Arb certification. Supplementary searches include polynomial curvature, signed higher harmonics, asymmetric phases, localized Fourier modes, convex quartic deformations, two interacting base frequencies, and positive-curvature lattices.

Resource sampling began at 2026-11-17T09:02:01.851026+00:00; 264 observations were recorded. Maximum observed host nonavailable-memory estimate: 6.10 GiB. Maximum sampled GPU memory use during that monitoring interval: 8,200 MiB. Earlier compiled-batch validation measured a 5.73-GB allocation. Sampling is not a continuous proof of a resource bound. Certification process pools were capped at four workers.

## Validation, limitations, and files

Formula tests at orders 6–8 passed the requested 1e-11 relative-error threshold against high-precision positive-time heat-convolution finite differences. Exact rational SOS identities were checked by a separate standard-library verifier. The counterexample has two different interval proof routes. Recovery checks exercised OOM handling and the bounded certification pool.

Several CPU jobs exited with code 143 without a Python traceback. Their saved results were preserved, incomplete runs were not represented as completed, and proof jobs were restarted with heartbeat/exit supervision. Numerical SDP failures and indefinite Gram matrices are explicitly not proofs of either positivity or failure. See [certs/process_interruptions.json](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/process_interruptions.json).

A finite search cannot prove a universal lower bound. The lower end of the bracket comes from exact certificates, not search minima. The remaining orders inside the bracket have not been resolved globally. No claim of novelty or peer review is made.

- [Research note PDF](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/paper/draft_note.pdf)
- [LaTeX source](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/paper/draft_note.tex)
- [CPU-only reproduction instructions](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/README.md)
- [Verified bracket and certificate hashes](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/checkpoints/research_status.json)
- [Formula validation](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/formula_tests.json)
- [Certified derivative table for the explicit witness](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/certs/witness_order_table.json)
- [Literature verification](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/notes_literature.md)
- [Progress and recorded counts](https://github.com/returnedlegend/gcm-logconcave-counterexamples/blob/main/PROGRESS.md)

Local working logs are in `logs/`; raw logs are excluded from Git. Source code, JSON checkpoints, proof certificates, the note, and this briefing are tracked.

Primary background: [Zou et al.](https://arxiv.org/abs/2608.30275), [Gu and Sellke](https://arxiv.org/abs/2605.11656), and [Wang](https://arxiv.org/abs/2408.13957).
