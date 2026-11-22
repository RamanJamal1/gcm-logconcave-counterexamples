# Authorship and AI-use disclosure

> **Before publishing:** adjust the description of the human contributions so it is
> accurate, and copy the disclosure paragraph (adapted to the venue's wording) into the paper. arXiv and most journals
> require AI use to be disclosed, and do not allow AI systems to be listed as authors.

## Author

- **Raman Jamal** (ORCID: [0009-0004-7042-8149](https://orcid.org/0009-0004-7042-8149)), the sole author and the person responsible for the
  content, its correctness and any submission.

## Contributions

- **Author:** posed and directed the research question; chose the research directions; supervised and reviewed the
  computations; is responsible for verifying the mathematics and for every claim made in any publication.
- **AI assistance:** large language models (Anthropic's Claude family, including Claude Opus and Sonnet models, run
  through Claude Code; earlier automated runs recorded in this repository as "GCM Research Automation") were used
  extensively to:
  - write and run code: symbolic derivations, GPU searches, semidefinite-programming solvers, exact rational
    reconstruction, and Arb interval-arithmetic certifiers;
  - draft mathematical arguments and documentation, including parts of `paper/`;
  - review and audit the code and certificates.
- Revisions for version 0.2.0 (review fixes: exact Zou-family rerun, hardened verifiers, Table 2 completion, the
  rewritten positive-time lemma and McKean-bound statements, references, repository clean-up) were made with Claude
  Code (Claude Opus 5.5).
- The unedited records of the automated runs (long-run prompt, run logs, briefings) are in
  [`archive/process-records/`](archive/process-records/README.md).
- AI systems are tools, not authors. They are not credited as authors, and no AI-generated claim is accepted without
  verification.

## What the verification does and does not cover

- The exact certificates (orders 6-11 and 13) are checked by a standalone verifier that uses only the Python standard
  library (`src/verify_sos_standalone.py`; it raises explicit errors, so it is safe under `python -O`, and
  `src/test_verifier_rejects.py` shows it rejecting corrupted certificates). The counterexamples are checked by two
  independent rigorous Arb interval computations.
- These checks establish the stated **algebraic identities and numerical enclosures**. The analytic steps that turn an
  identity into an inequality for every log-concave density (Lemma 3.3 on positive-time regularity, the endpoint
  argument at t = 0, the downward-propagation lemma and the proof of Theorem 1.4) are written arguments in `paper/`.
  They need **independent checking by a human mathematician** before publication.

## Disclosure paragraph (as used in the paper; adapt to the venue)

> This work was carried out with AI systems doing most of the technical work. Anthropic's Claude models, run through
> Claude Code (including long autonomous runs coordinated by a Claude model), proposed the computational strategy,
> wrote and ran all of the software (symbolic derivations, numerical searches, semidefinite solvers, exact rational
> reconstruction and the interval-arithmetic certifiers), found the certificates and the counterexamples, and drafted
> the mathematical arguments and the text. The revisions for version 0.2.0 were also made with Claude Code. The author
> posed the question, directed and supervised the work, and reviewed the results. All certificates and enclosures are
> checked by the exact rational and interval-arithmetic verifiers released with the paper; the analytic arguments of
> Sections 3.2–3.5 are written proofs that have not yet been independently checked by a human expert. The unedited
> records of the automated runs are archived with the code. The author takes full responsibility for the content.
>
> Once a human expert has checked Sections 3.2–3.5, replace "have not yet been independently checked by a human
> expert" with an accurate description of who checked them.
