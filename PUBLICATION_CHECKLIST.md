# Publication checklist

A practical list of what remains before this work can be published under your name. Items marked **(you)** need you
personally; the rest can be prepared in the repository.

## 1. Authorship and ownership
- [x] Author name (Raman Jamal) set in `CITATION.cff`, `.zenodo.json`, `AUTHORSHIP_AND_AI_DISCLOSURE.md` and `LICENSE`.
- [x] ORCID 0009-0004-7042-8149 added to `CITATION.cff`, `.zenodo.json` and `AUTHORSHIP_AND_AI_DISCLOSURE.md`.
- [ ] **(you)** Decide whether to keep the repository public before submission. Public code is fine, and many venues
      encourage it, but it also counts as prior disclosure.

## 2. Is it new?
- [x] Literature search (2026-11-21): no later work found that fixes the first failing order for log-concave densities,
      the Zou family's threshold, or gives SOS certificates at orders ≥ 6. Prior work now cited: Zhang–Anantharam–Geng
      (Entropy 2018, the LMI method and McKean's bound up to order 5), Wang (IEEE TIT 2025), Guo–Yuan–Gao (Entropy 2022),
      Gu–Sellke, Zou–Fan–Gao–Wang (both papers). Repeat the search just before submission.

## 3. Is it correct? (most important)
- [ ] **(you)** Independent human review of the analytic arguments: Lemma 3.3 (positive-time regularity: bounded
      log-derivatives via the conditional law, integration by parts, differentiation of the entropy), the endpoint
      argument at t = 0, the downward-propagation lemma and the proof of Theorem 1.4 (paper Sections 3.2–3.5).
      When done, update the "Use of AI" paragraph in the paper and `AUTHORSHIP_AND_AI_DISCLOSURE.md`.
- [x] All verifiers re-run from a fresh clone on 2026-11-21 (Windows 11, Python 3.13, python-flint 0.9.0): all 13 exact
      certificates pass under both `python` and `python -O`; both order-16 enclosures, the curvature proof, the Arb
      order-13 check and the log-cosh family reproduce.
- [x] Verifiers no longer rely on `assert` (safe under `python -O`); `src/test_verifier_rejects.py` checks that
      corrupted certificates are rejected; the sharp-certificate subtraction is now checked, not trusted.
- [x] Zou-family threshold recomputed for the exact family (Arb-ball parameters, no member skipped), with an independent
      cross-check at M = 32, 33, 36.
- [ ] Ideally, have a second person (or a second implementation in another computer-algebra system) re-check the
      certificates.

## 4. The paper
- [x] `paper/main.tex` revised to version 0.2.0 (review fixes; see the git log of branch `review-fixes`).
- [x] `paper/main.pdf` rebuilt from the v0.2.0 source on 2026-11-21 (`tectonic -X compile paper/main.tex`, Tectonic
      0.17.0; 13 pages, no undefined references, no overfull lines).
- [x] Table 2 completed (orders 9–11 added); Theorem 1.6 (Zou family) now rigorous for the exact family.
- [x] AI-use statement rewritten to describe the actual extent of AI work; data/code availability statement updated.
- [ ] Add the archived release DOI to the data/code availability statement (after Zenodo, see §6).
- [ ] Decide the headline result. Currently: the bracket **14 ≤ m* ≤ 16**, McKean's bound for log-concave densities
      through order 11, and the certified Zou-family threshold **33**. An order-15 certificate would make it **m* = 16**.

## 5. Third-party material and privacy
- [x] Copies of other authors' papers were removed from `literature/` and replaced by citations and links
      (`literature/README.md`).
- [x] A second GitHub account name that appeared in `certs/preflight.json`, `certs/gpu_runtime_check.json`,
      `certs/latest_sync.json` and two process records was redacted.
- [ ] **(you)** The git history still contains the redacted account name and the removed third-party copies. Rewrite the
      history before archiving only if that matters to you (it is destructive and changes every commit hash).

## 6. Archiving and citation
- [x] `.gitattributes` keeps certificate bytes identical on Windows; `certs/exact_certificate_sha256.json` covers every
      cited certificate and enclosure (`src/check_hashes.py`); `requirements-verify.txt` lists the only dependency.
- [x] AI-run process records moved to `archive/process-records/`.
- [ ] **(you)** Connect the GitHub repository to Zenodo (https://zenodo.org → GitHub integration), then create a GitHub
      release; Zenodo then issues a DOI using `.zenodo.json`.
- [ ] Put the DOI into `CITATION.cff` and the paper's code-availability statement.

## 7. Submission
- [ ] **(you)** arXiv: create an account; a first submission in a category may need an endorsement. Likely categories:
      math.PR (primary) with cs.IT / math.FA as cross-lists.
- [ ] **(you)** Journal: IEEE Transactions on Information Theory (where Cheng–Geng and Wang appeared) or *Experimental
      Mathematics*; check each journal's AI and data policies.
- [ ] **(you)** Sign the venue's copyright or licence agreement. Keep the code under MIT and the data under CC BY 4.0;
      most journals accept that, as long as the article itself follows the journal's terms.
