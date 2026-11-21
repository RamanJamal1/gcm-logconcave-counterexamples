# RUNLOG (append-only)

## 2026-11-21T21:28:44Z — start-up
- Cloned repo, switched to research/order15-mechanisms (HEAD aa00466). Read HANDOFF_2026-11-21.md.
- Bracket: 14 <= m* <= 16. Goal: exact rational SOS certificate for S_15 >= 0.
- Delegated: Haiku summary of tracks (notes/tracks_summary.md); Sonnet env setup + prep_sparse for m=13,14,15 (tracks/setup/).
- Launched X1 (Opus subagent): verify positive-time compactification & explicit C_k (tracks/X1_compactification/).
- 2026-11-21T21:28:57Z BLOCKER: git push returns HTTP 403 (fine-grained PAT lacks contents:write). Commits kept locally; user must fix token. Will retry periodically.
- 2026-11-21T21:29:52Z Audit: verifier accepts arbitrary target_subtraction (could subtract S_m itself). Added tracks/director/verify_gate.py: requires none, order match, -O flag. m6 passes; tampered (subtraction u2**6) rejected. Running gate on existing m13 cert (timing baseline).
- 2026-11-21T21:31:25Z Setup done (tracks/setup/SETUP.md): m13 478 cons/101+77, m14 708/135+101, m15 1039/176+135; torch CUDA, flint, sdpap OK; strict verifier accepts m6. Haiku summary in ~/gcm-run/notes/tracks_summary.md (E build_dd hard-codes 1e-5 margin; F/G never run).
- 2026-11-21T21:31:25Z Launched P1 (Sonnet, GPU): order-13 end-to-end, margin 0, analytic center seed -> G pipeline -> gate. tracks/P1_order13/.
- 2026-11-21T21:31:25Z Launched X3 (Sonnet, CPU<=4 cores): SDPA-GMP max-margin at m=13,14,15; save dual if m15 negative. tracks/X3_maxmargin/.
- 2026-11-21T21:33:51Z X1 DONE (tracks/X1_compactification/REPORT.md): Tweedie identities, -1<u2<=0, explicit C_k (C_3=4.51..C_16=6.8e13, crude, Arb) VERIFIED. Thm 5: enlarged compactified ansatz gives NO gain for single-order targets (lowest-weight component of any such certificate is already a certificate of the existing homogeneous form). Director review: argument sound (weight grading preserved by IBP; lowest-weight part of SOS is SOS; IBP-exact nonzero SOS impossible). DECISION: drop enlarged-ansatz fallback; if m15 max margin < 0, the only routes are new weight-homogeneous jet inequalities valid for log-concave p, or the counterexample hedge (X5).
- Push works again (403 resolved).
