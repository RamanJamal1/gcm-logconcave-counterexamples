# Publication checklist

A practical list of what remains before this work can be published under your name. Items marked **(you)** need you
personally; the rest can be prepared in the repository.

## 1. Authorship and ownership
- [x] Author name (Raman Jamal) set in `CITATION.cff`, `.zenodo.json`, `AUTHORSHIP_AND_AI_DISCLOSURE.md` and `LICENSE`.
- [ ] **(you)** Add your ORCID to `CITATION.cff` and `.zenodo.json` (free at https://orcid.org).
- [ ] **(you)** Decide whether to keep the repository public before submission. Public code is fine, and many venues
      encourage it, but it also counts as prior disclosure.

## 2. Is it new?
- [ ] Systematic literature search after the two 2026 papers (arXiv:2605.11656, arXiv:2608.30275) for any result on the
      first failing order for log-concave densities, the Zou family's threshold, or exact SOS certificates of this kind.
      The repository currently makes **no novelty claim**.

## 3. Is it correct? (most important)
- [ ] Independent human review of the analytic hypotheses: integration by parts with vanishing boundary terms for
      log-concave densities, the positive-time smoothing and t → 0 endpoint argument, and the downward-propagation
      lemma (`paper/analytic_scope.tex`, `certs/downward_propagation.md`).
- [ ] Re-run every verifier from a fresh clone on a clean machine (README "Reproduce the proofs" section), record
      the versions and runtimes, and **do not use `python -O`** with `src/verify_sos_standalone.py`.
- [ ] Ideally, have a second person (or a second independent implementation) re-check the certificates.

## 4. The paper
- [x] `paper/draft_note.pdf` rebuilt from the current `.tex` (2026-11-21, Tectonic 0.17.0; a fragile `\path` inside a caption was fixed).
- [ ] Revise the draft's framing: it is titled a "computational research record", marked "provisional during computation", and
      has no author line. Add **Raman Jamal** as author and rewrite the abstract for the venue.
- [ ] Decide the headline result. With the current state: the bracket **14 ≤ m* ≤ 16** with explicit certificates and
      witness, plus the certified Zou-family threshold **m₀ = 33**. If an order-15 certificate is later verified, the
      headline becomes **m* = 16**.
- [ ] Add the AI-use disclosure (see `AUTHORSHIP_AND_AI_DISCLOSURE.md`), a data/code availability statement pointing to
      the archived release (DOI), and acknowledgements.
- [ ] Remove internal or automation language from the text (for example "campaign", "briefing", automated status notes).

## 5. Third-party material
- [x] Copies of other authors' papers were removed from `literature/` and replaced by citations and links
      (`literature/README.md`). They remain in the git history; if that matters, the history can be rewritten.

## 6. Archiving and citation
- [ ] **(you)** Connect the GitHub repository to Zenodo (https://zenodo.org → GitHub integration), then create a GitHub
      release; Zenodo then issues a DOI using `.zenodo.json`.
- [ ] Put the DOI into `CITATION.cff` and the paper's code-availability statement.

## 7. Submission
- [ ] **(you)** arXiv: create an account; a first submission in a category may need an endorsement. Likely categories:
      math.PR (primary) with cs.IT / math.FA / math.CA as cross-lists.
- [ ] **(you)** Journal: computer-assisted results of this type commonly go to venues such as *Experimental Mathematics*,
      *Mathematics of Computation*, or information-theory and probability journals; check each journal's AI and
      data policies.
- [ ] **(you)** Sign the venue's copyright or licence agreement. Keep the code under MIT and the data under CC BY 4.0;
      most journals accept that, as long as the article itself follows the journal's terms.
