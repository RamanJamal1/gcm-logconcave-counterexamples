# Log-concave entropy derivatives: computational research record

This repository studies the signs of higher time-derivatives of entropy along the heat flow for log-concave densities. With `S_m(p) = (-1)^m d^m/dt^m H(P_t p)` at `t = 0`, the question is the smallest order `m*` at which some log-concave density has `S_m < 0`. This is the open question left by Gu–Sellke (arXiv:2605.11656) and Zou–Fan–Gao–Wang (arXiv:2608.30275).

The current verified bracket is **14 ≤ m* ≤ 16**, with the endpoint regularity made explicit in the note. The exact minimal order remains unresolved: orders 14 and 15 have neither a certificate nor a counterexample.

**Status (2026-11-21).** The original search campaigns finished on 2026-11-17; [the briefing](MORNING_BRIEFING.md) and [progress](PROGRESS.md) are records of that run, and their "running" wording is historical. Follow-up work on orders 14–15 lives on the branch [`research/order15-mechanisms`](https://github.com/RamanJamal1/gcm-logconcave-counterexamples/tree/research/order15-mechanisms); see its `HANDOFF_2026-11-21.md`. That branch contains exploratory numerics and new solver code but **no new certificate**. All numerical evidence so far (billions of candidate evaluations over several density families) is consistent with `m* = 16`; it is evidence, not proof.

The convention is Boltzmann entropy `H(p) = integral p log p` and heat generator `(1/2) d²/dx²`. Reversing the entropy convention reverses the derivative signs.

## Verified results

- Exact rational sum-of-squares identities establish nonnegativity at orders 6–11 and 13 under justified integration by parts. Order 12 follows by downward propagation along heat flow. The note gives the positive-time smoothing and endpoint argument.
- Stronger identities give `S_m Var(p)^m ≥ (m-1)!/2` for every order 6–11, including the sharp bounds `S6 Var^6 ≥ 60`, `S7 Var^7 ≥ 360`, and `S8 Var^8 ≥ 2520`.
- The explicit density

  ```text
  p(x) = exp(-x²/28)
         * (1 + cos(x)/15 + cos(2*x)/40000 - cos(3*x)/18000000) / Z
  ```

  has `V'' > 1/10000` globally and `S16 < -3/4`. Here

  ```text
  Z = sqrt(28*pi)
      * (1 + exp(-7)/15 + exp(-28)/40000
           - exp(-63)/18000000).
  ```

  Two independent Arb calculations certify its signed derivative near `-0.76971250884778794609`: [spatial score-polynomial integration](certs/paired15_m16.json) and [exact heat-density Taylor integration](certs/independent_paired15_m16.json).

- The explicit family of Zou–Fan–Gao–Wang, `f_m ∝ exp(-κx²/2)(1 + a cos x + b cos 2x)` with `a = 2^(-m/4)`, `b = 2^(-m)`, `κ = 8a`, which their paper shows fails only for "sufficiently large m", first fails at order **33**: rigorous Arb series enclosures give `S_M(f_M) > 0` for `6 ≤ M ≤ 32` and `S_M(f_M) < 0` for every evaluated `33 ≤ M ≤ 400` ([data](checkpoints/zou_family_threshold.json), [script](src/zou_threshold.py)).

Read [the LaTeX note](paper/draft_note.tex) for definitions, proofs, analytic error bounds, and limitations. **The compiled [PDF](paper/draft_note.pdf) is out of date**: it predates the current `.tex` and still shows an older bracket and witness, so treat the `.tex` source as authoritative until the PDF is rebuilt. No novelty or exact-minimality claim is made.

## Reproduce the proofs on a CPU

Run from the repository root. The rational SOS verifier uses only the Python standard library:

```bash
python3 src/verify_sos_standalone.py \
  certs/exact_sos_m6.json certs/exact_sos_m7.json certs/exact_sos_m8.json \
  certs/exact_quotient_m9.json certs/exact_quotient_m10.json \
  certs/exact_quotient_m11.json certs/exact_quotient_m13.json \
  certs/exact_mckean_m6.json certs/exact_mckean_m7.json certs/exact_mckean_m8.json \
  certs/exact_mckean_quotient_m9.json certs/exact_mckean_quotient_m10.json \
  certs/exact_mckean_quotient_m11.json
```

It independently derives the entropy polynomial, checks each integration-by-parts identity, and proves positivity of the rational Gram matrices by exact Schur complements. A numerical SDP status is never used as a proof.

> **Do not run this verifier with `python -O`.** It reports failures through `assert` statements, which `-O` removes, so under `-O` it would accept invalid certificates. Run it with plain `python3` as shown. The research branch adds `tracks/D_order14/verify_strict.py`, which runs the same verifier with every `assert` rewritten into an explicit error, so it is also valid under `-O`.
>
> Running the verifier's command-line entry point also writes `certs/independent_sos_verification.json`.

The exact global minimum curvature is `25777/235206314 > 1/10000`; its independent standard-library proof is reproduced with `python3 src/certify_min_curvature.py`.

The entropy-derivative verification needs `python-flint`:

```bash
python3 -m venv .venv
.venv/bin/pip install python-flint==0.9.0
.venv/bin/python src/certify_multifourier_spatial.py
.venv/bin/python src/certify_direct_heat.py certs/paired15_m16.json \
  --output certs/independent_paired15_m16.json
```

The standard-library order-13 check took about 31 minutes in this run. For that larger identity, a faster independent matrix check uses exact rational polynomial arithmetic and outward-rounded Arb Schur complements:

```bash
.venv/bin/python src/verify_sos_arb.py certs/exact_quotient_m13.json
.venv/bin/python src/verify_logcosh_family.py
```

The second command checks the entire tilted hyperbolic-secant family at orders 2–17 using exact beta-moment and Bernstein identities. It is a family-specific exclusion, not a universal higher-order sign.

Both routines use 200-digit ball arithmetic with analytic bounds for the omitted infinite tails. High precision alone is not treated as a rigorous enclosure. The independent calculation differentiates the exact heat evolution and bounds its tail using Cauchy's estimate.

## Search and validation

`requirements-research.txt` records the broader research environment. GPU searches use PyTorch/CUDA, FP32 exploration, and FP64 refinement with a separate denser-grid check. Production states are in `checkpoints/campaign_A.json`, `campaign_B.json`, and `campaign_C.json`. Note that `src/campaign_c.py` and `src/campaign_c_multimode.py` both default to `checkpoints/campaign_C.json`, and `campaign_c.py` does not resume from it, so pass a separate `--state` path when running it. Supplementary searches and readiness runs have separate state files. Counts are candidate evaluations, not unique probability densities.

- `src/derive_integrands.py` and `src/test_formulas.py`: symbolic formulas and independent high-precision heat-convolution finite differences at orders 6–8.
- `src/entropy_sparse.py`: faster exact rational recurrence for higher orders.
- `src/sos_*.py`: numerical searches for identities; their outputs marked `proved: false` are not certificates.
- `src/reconstruct_quotient.py`: exact rational reconstruction, followed by independent verification.
- `src/certify_fourier_series.py`: an additional rigorous Fourier-series method, used for the independently checked order-17 example.
- `src/certify_multifourier_spatial.py` and `src/certify_direct_heat.py`: independent order-16 proofs.
- `src/checkpoint.py`: serialized Git checkpoints. It runs `git add -A`, commits and pushes to `origin main`, and campaigns call it when started with `--sync`.

The original papers and source retrieval records are under `literature/`; [literature notes](notes_literature.md) record the conventions and verified source claims. A finite unsuccessful search does not establish universal positivity. The interval `[14,16]` combines exact lower-order identities with a certified upper-bound example, not empirical minima.

## License

- **Code** (`src/` and the scripts under `tracks/` on the research branch): [MIT License](LICENSE).
- **Paper, documentation, certificates and research data** (`paper/`, the Markdown files, `certs/`, `checkpoints/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt).
- **Third-party material** under `literature/` (copies of published papers and their sources) is covered by neither license and remains under its authors' original terms.
