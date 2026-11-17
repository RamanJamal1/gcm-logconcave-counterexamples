# Log-concave entropy derivatives: computational research record

The current verified bracket is **12 ≤ m* ≤ 16**, with the endpoint regularity made explicit in the note. The exact minimal order remains unresolved. The scheduled overnight campaign is still running; see [the briefing](MORNING_BRIEFING.md) and [progress](PROGRESS.md).

The convention is Boltzmann entropy `H(p) = integral p log p` and heat generator `(1/2) d²/dx²`. Reversing the entropy convention reverses the derivative signs.

## Verified results

- Exact rational sum-of-squares identities establish nonnegativity at orders 6–11 under justified integration by parts. The note gives the positive-time smoothing and endpoint argument.
- Stronger identities give `S_m Var(p)^m ≥ (m-1)!/2` for every order 6–11, including the sharp bounds `S6 Var^6 ≥ 60`, `S7 Var^7 ≥ 360`, and `S8 Var^8 ≥ 2520`.
- The explicit density

  ```text
  p(x) = exp(-x²/30)
         * (1 + cos(x)/16 + cos(2*x)/48000 - cos(3*x)/20000000) / Z
  ```

  has `V'' > 1/12000` globally and `S16 < -3/4`. Here

  ```text
  Z = sqrt(30*pi)
      * (1 + exp(-15/2)/16 + exp(-30)/48000
           - exp(-135/2)/20000000).
  ```

  Two independent Arb calculations certify its signed derivative near `-0.76099654207878681651`: [spatial score-polynomial integration](certs/simple_kappa_48000_m16.json) and [exact heat-density Taylor integration](certs/independent_simple_kappa_48000_m16.json).

Read [the LaTeX note](paper/draft_note.tex) or [its PDF](paper/draft_note.pdf) for definitions, proofs, analytic error bounds, and limitations. No novelty or exact-minimality claim is made.

## Reproduce the proofs on a CPU

Run from the repository root. The rational SOS verifier uses only the Python standard library:

```bash
python3 src/verify_sos_standalone.py \
  certs/exact_sos_m6.json certs/exact_sos_m7.json certs/exact_sos_m8.json \
  certs/exact_quotient_m9.json certs/exact_quotient_m10.json \
  certs/exact_quotient_m11.json \
  certs/exact_mckean_m6.json certs/exact_mckean_m7.json certs/exact_mckean_m8.json \
  certs/exact_mckean_quotient_m9.json certs/exact_mckean_quotient_m10.json \
  certs/exact_mckean_quotient_m11.json
```

It independently derives the entropy polynomial, checks each integration-by-parts identity, and proves positivity of the rational Gram matrices by exact Schur complements. A numerical SDP status is never used as a proof.

The exact global minimum curvature is `76658/843768795 > 1/12000`; its independent standard-library proof is reproduced with `python3 src/certify_min_curvature.py`.

The entropy-derivative verification needs `python-flint`:

```bash
python3 -m venv .venv
.venv/bin/pip install python-flint==0.9.0
.venv/bin/python src/certify_multifourier_spatial.py
.venv/bin/python src/certify_direct_heat.py certs/simple_kappa_48000_m16.json \
  --output certs/independent_simple_kappa_48000_m16.json
```

Both routines use 200-digit ball arithmetic with analytic bounds for the omitted infinite tails. High precision alone is not treated as a rigorous enclosure. The independent calculation differentiates the exact heat evolution and bounds its tail using Cauchy's estimate.

## Search and validation

`requirements-research.txt` records the broader research environment. GPU searches use PyTorch/CUDA, FP32 exploration, and FP64 refinement with a separate denser-grid check. Production states are in `checkpoints/campaign_A.json`, `campaign_B.json`, and, after it starts, `campaign_C.json`. Supplementary searches and readiness runs have separate state files. Counts are candidate evaluations, not unique probability densities.

- `src/derive_integrands.py` and `src/test_formulas.py`: symbolic formulas and independent high-precision heat-convolution finite differences at orders 6–8.
- `src/entropy_sparse.py`: faster exact rational recurrence for higher orders.
- `src/sos_*.py`: numerical searches for identities; their outputs marked `proved: false` are not certificates.
- `src/reconstruct_quotient.py`: exact rational reconstruction, followed by independent verification.
- `src/certify_fourier_series.py`: an additional rigorous Fourier-series method, used for the independently checked order-17 example.
- `src/certify_multifourier_spatial.py` and `src/certify_direct_heat.py`: independent order-16 proofs.
- `src/checkpoint.py`: serialized Git checkpoints and private GitHub backup.

The original papers and source retrieval records are under `literature/`; [literature notes](notes_literature.md) record the conventions and verified source claims. A finite unsuccessful search does not establish universal positivity. The interval `[12,16]` combines exact lower-order identities with a certified upper-bound example, not empirical minima.
