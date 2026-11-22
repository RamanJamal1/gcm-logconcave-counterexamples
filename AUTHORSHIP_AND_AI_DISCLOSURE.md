# Authorship and AI-use disclosure

> **TODO before publishing:** fill in the placeholders, adjust the description of the human contributions so it is
> accurate, and copy the disclosure paragraph (adapted to the venue's wording) into the paper. arXiv and most journals
> require AI use to be disclosed, and do not allow AI systems to be listed as authors.

## Author

- **<GIVEN NAME> <FAMILY NAME>** (ORCID: 0000-0000-0000-0000), the sole author and the person responsible for the
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
- AI systems are tools, not authors. They are not credited as authors, and no AI-generated claim is accepted without
  verification.

## What the verification does and does not cover

- The exact certificates (orders 6-11 and 13) are checked by a standalone verifier that uses only the Python standard
  library (`src/verify_sos_standalone.py`; run it without `python -O`). The counterexamples are checked by two
  independent rigorous Arb interval computations.
- These checks establish the stated **algebraic identities and numerical enclosures**. The analytic steps that turn an
  identity into an inequality for every log-concave density (integration by parts with vanishing boundary terms,
  the endpoint/Taylor argument at t = 0, and the downward-propagation lemma) are written arguments in `paper/` and
  `certs/downward_propagation.md`. They need **independent checking by a human mathematician** before publication.

## Suggested disclosure paragraph (adapt to the venue)

> This work made extensive use of AI assistants (Anthropic's Claude models, via Claude Code) to write and run the
> software, including the symbolic computations, numerical searches and semidefinite solvers that produced the
> certificates and counterexamples, and to help draft parts of the text. All certificates are verified by
> independent exact rational or rigorous interval-arithmetic checkers included with the paper. The author checked
> the mathematical arguments and takes full responsibility for the content.
