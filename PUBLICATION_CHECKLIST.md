# Publication checklist

A practical list of what remains before this work can be published under your name. Items marked **(you)** need you
personally; the rest can be prepared in the repository.

## 1. Authorship and ownership
- [ ] **(you)** Fill in your real name and ORCID in `CITATION.cff`, `.zenodo.json` and `AUTHORSHIP_AND_AI_DISCLOSURE.md`
      (get a free ORCID at https://orcid.org if you don't have one).
- [ ] **(you)** Decide the copyright holder name in `LICENSE` (currently "returnedlegend and contributors").
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
- [ ] Rebuild `paper/draft_note.pdf` from the current `.tex` (the committed PDF is out of date).
- [ ] Decide the headline result. With the current state: the bracket **14 ≤ m* ≤ 16** with explicit certificates and
      witness, plus the certified Zou-family threshold **m₀ = 33**. If an order-15 certificate is later verified, the
      headline becomes **m* = 16**.
- [ ] Add the AI-use disclosure (see `AUTHORSHIP_AND_AI_DISCLOSURE.md`), a data/code availability statement pointing to
      the archived release (DOI), and acknowledgements.
- [ ] Remove internal or automation language from the text (for example "campaign", "briefing", automated status notes).

## 5. Third-party material
- [ ] `literature/` contains copies of other authors' papers and their source files. These are **not** covered by this
      repository's licenses, and redistributing them publicly may not be permitted. Before publication, consider
      replacing them with links and citations (the arXiv identifiers are in `notes_literature.md`).

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
