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
- 2026-11-21T21:42:45Z RESTART. INBOX directive: skip Phase 1, only critical job = m15 max margin; no X2/X4/X6/X7; X5 only if margin<0. P1/X3 agents were killed by overseer.
- X3 state: SDPA-GMP m13 diverging (free-variable split, objP 5e6), in-house ipm.py crashed (Cholesky). Not reused.
- DECISION: decide m15 with the E_ddipm double-double dual-barrier solver (converged at m13 with delta=0). Copied to tracks/M15/ (output dir changed). Job: build_dd 15 && solve.py 15 --delta 0 (PID 9106, log tracks/M15/m15.log). Convergence to an analytic center => strict interior exists (margin > 0); divergence of -b^T y => dual ray (margin <= 0).
- 2026-11-21T21:58:22Z M15 poll: dd it75 (860 s, 11.6 s/it). |g|inf 431->182 slowly falling, lam2 stuck ~3.05 at alpha=1, phi 1373->1367 (~-2/it), Xmin ~4.36e-5/5.03e-5, Xmax 3e7->1.3e7. Undecided; continue to maxit 300.
- 2026-11-21T22:19:29Z M15 it186: |g| 93.6 (slowing), lam2 3.05, Xmin block0 4.4e-5 -> 9.2e-6 FALLING (approaching boundary), block1 ~5.1e-5. Per overseer: snapshot ck at it~185 (data/ck_snapA.npz); dual-ray screen tracks/M15/raytest.py on d=y_k-y_{k-50} next poll; resume past maxit 300.
