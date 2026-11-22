# Log-concave entropy derivatives: computational research record

This repository studies the signs of higher time-derivatives of entropy along the heat flow for log-concave densities. With `S_m(p) = (-1)^m d^m/dt^m H(P_t p)` at `t = 0`, the question is the smallest order `m*` at which some log-concave density has `S_m < 0`. This is the open question left by Gu–Sellke (arXiv:2605.11656) and Zou–Fan–Gao–Wang (arXiv:2608.30275).

The current verified bracket is **14 ≤ m* ≤ 16**, with the endpoint regularity made explicit in the note. The exact minimal order remains unresolved: orders 14 and 15 have neither a certificate nor a counterexample.

**Status (2026-11-21, version 0.2.0).** The original search campaigns finished on 2026-11-17. The unedited records of those AI-driven runs (prompt, run logs, [briefing](archive/process-records/MORNING_BRIEFING.md), [progress](archive/process-records/PROGRESS.md)) are kept in [`archive/process-records/`](archive/process-records/README.md); their "running" wording is historical. Follow-up work on orders 14–15 lives on the branch [`research/order15-mechanisms`](https://github.com/RamanJamal1/gcm-logconcave-counterexamples/tree/research/order15-mechanisms); see its `HANDOFF_2026-11-21.md`. That branch contains exploratory numerics and new solver code but **no new certificate**. All numerical evidence so far (billions of candidate evaluations over several density families) is consistent with `m* = 16`; it is evidence, not proof.

The convention is Boltzmann entropy `H(p) = integral p log p` and heat generator `(1/2) d²/dx²`. Reversing the entropy convention reverses the derivative signs.

## Verified results

- An exact rational sum-of-squares identity at order 13, plus downward propagation along the heat flow, gives `S_m(P_t p) ≥ 0` for every log-concave `p`, every `t > 0` and every `m ≤ 13` (direct identities also exist at orders 6–11). At `t = 0` the sign follows under the endpoint-regularity assumption stated in the paper.
- Stronger identities give McKean's Gaussian optimality bound `S_m Var(p)^m ≥ (m-1)!/2` for log-concave densities at every order `m ≤ 11` (previously known for `m ≤ 5`), including `S6 Var^6 ≥ 60`, `S7 Var^7 ≥ 360`, and `S8 Var^8 ≥ 2520`.
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

- The explicit family of Zou–Fan–Gao–Wang, `f_m ∝ exp(-κx²/2)(1 + a cos x + b cos 2x)` with `a = 2^(-m/4)`, `b = 2^(-m)`, `κ = 8a`, which their paper shows fails only for "sufficiently large m", first fails at order **33**: rigorous Arb series enclosures for the exact members (`a = 2^(-M/4)` enclosed in an Arb ball; no member skipped or approximated) give `S_M(f_M) > 0` for `6 ≤ M ≤ 32` and `S_M(f_M) < 0` for `33 ≤ M ≤ 400` ([data](checkpoints/zou_family_threshold.json), [script](src/zou_threshold.py), about 3.5 minutes). `src/zou_crosscheck.py` confirms `M = 32, 33, 36` with the independent heat-evolution method ([data](checkpoints/zou_crosscheck.json)).
- Table 2 of the paper (certified `S_m` of the order-16 witness) now covers every order 6–20; the per-order enclosures are in `certs/witness_order_table_paired15/`.

**Paper:** [`paper/main.pdf`](paper/main.pdf) ([LaTeX source](paper/main.tex)) is the structured manuscript prepared for expert review: main theorems, the certificate method, the analytic arguments, the order-16 counterexample, the Zou-family threshold, and a discussion of orders 14–15. Section 1 states explicitly which steps are machine-checked and which are written arguments needing review. The original research note, [`paper/draft_note.pdf`](paper/draft_note.pdf) ([source](paper/draft_note.tex)), is kept as the detailed computational record.

Read the paper for definitions, proofs, analytic error bounds, and limitations. Build the PDF with `tectonic -X compile paper/main.tex` (Tectonic 0.17.0). The analytic arguments (Lemma 3.3 and Sections 3.2–3.5) have not yet been checked by an independent human expert.

## Reproduce the proofs on a CPU

Run from the repository root (use `python` instead of `python3` on Windows). On Windows, clone with `git -c core.autocrlf=false clone ...` so certificate bytes match the hash manifest; `.gitattributes` also protects `certs/` and `checkpoints/`. First check that the data are the published bytes:

```bash
python3 src/check_hashes.py
```

The rational SOS verifier uses only the Python standard library:

```bash
python3 src/verify_sos_standalone.py \
  certs/exact_sos_m6.json certs/exact_sos_m7.json certs/exact_sos_m8.json \
  certs/exact_quotient_m9.json certs/exact_quotient_m10.json \
  certs/exact_quotient_m11.json certs/exact_quotient_m13.json \
  certs/exact_mckean_m6.json certs/exact_mckean_m7.json certs/exact_mckean_m8.json \
  certs/exact_mckean_quotient_m9.json certs/exact_mckean_quotient_m10.json \
  certs/exact_mckean_quotient_m11.json
```

It independently derives the entropy polynomial, checks each integration-by-parts identity, checks that every sharp certificate subtracts exactly `(m-1)!/2 (-u2)^m`, and proves positivity of the rational Gram matrices by exact Schur complements. A numerical SDP status is never used as a proof. It prints one JSON line per certificate, exits non-zero if any certificate fails, and writes a results file only when asked (`--output PATH`).

All verifiers raise explicit errors instead of using `assert`, so they are equally strict under `python -O`. To see the verifier reject corrupted certificates (perturbed Gram entry, wrong IBP multiplier, non-positive-definite or asymmetric Gram, wrong sharp term, dropped IBP term) under both `python` and `python -O`:

```bash
python3 src/test_verifier_rejects.py
```

The exact global minimum curvature is `25777/235206314 > 1/10000`; its independent standard-library proof is reproduced with `python3 src/certify_min_curvature.py --output /tmp/curvature.json` (without `--output` it rewrites the tracked file).

The entropy-derivative verification needs `python-flint` (the only entry in `requirements-verify.txt`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-verify.txt
.venv/bin/python src/certify_multifourier_spatial.py
.venv/bin/python src/certify_direct_heat.py certs/paired15_m16.json \
  --output certs/independent_paired15_m16.json
.venv/bin/python src/zou_threshold.py      # Zou family, M = 6..400 (about 3.5 minutes)
.venv/bin/python src/zou_crosscheck.py     # M = 32, 33, 36 by the heat-evolution method
```

The standard-library order-13 check takes 15–30 minutes on one core, almost all of it in the exact Schur complements. For that larger identity, a faster independent matrix check uses exact rational polynomial arithmetic and outward-rounded Arb Schur complements:

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
- `src/certify_fourier_series.py`: an additional rigorous Fourier-series method, used for the independently checked order-17 example and the Zou family; `certify_balls` accepts Arb-ball parameters.
- `src/certify_multifourier_spatial.py` and `src/certify_direct_heat.py`: independent order-16 proofs.
- `src/checkpoint.py`: serialized Git checkpoints. It runs `git add -A`, commits and pushes to `origin main`, and campaigns call it when started with `--sync`.

Citations and links for the two source papers are in [`literature/README.md`](literature/README.md) (the downloaded copies were removed for licensing reasons); [literature notes](notes_literature.md) record the conventions and verified source claims. A finite unsuccessful search does not establish universal positivity. The interval `[14,16]` combines exact lower-order identities with a certified upper-bound example, not empirical minima.

## License

- **Code** (`src/` and the scripts under `tracks/` on the research branch): [MIT License](LICENSE).
- **Paper, documentation, certificates and research data** (`paper/`, the Markdown files, `certs/`, `checkpoints/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-CC-BY-4.0.txt).
- **Third-party works** cited here (the papers listed in `literature/README.md`) are not included in the repository and remain under their authors' terms.

## Citing and publishing

Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub shows a "Cite this repository" button from it. [`.zenodo.json`](.zenodo.json) holds metadata for an archived, DOI-bearing release. Authorship and the disclosure of AI assistance are described in [`AUTHORSHIP_AND_AI_DISCLOSURE.md`](AUTHORSHIP_AND_AI_DISCLOSURE.md), and the remaining steps toward publication are in [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
