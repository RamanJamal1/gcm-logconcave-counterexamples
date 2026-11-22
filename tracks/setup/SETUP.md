# Setup report
Python 3.12.13, A100-SXM4-80GB. Installed (user site) via `pip install -r requirements.txt -r requirements-research.txt` (~50 s, log pip.log).
Versions: numpy 2.5.0, scipy 1.18.0, sympy 1.14.0, mpmath 1.3.0, python-flint 0.9.0, torch 2.12.1+cu130 (CUDA 13.0), cvxpy 1.9.3, sdpa-multiprecision 0.2.3 (sdpap).
nvcc: not installed; grep for nvcc/cpp_extension/RawKernel in tracks/ and src/ found nothing.

## Data (python3 tracks/D_order14/prep_sparse.py M; written to tracks/D_order14/data/, untracked, *.npz gitignored, NOT committed)
| m | rows(constraints) | cols | nnz | Gram blocks | time |
|---|---|---|---|---|---|
| 13 | 478 | 16130 | 1305465 | 101+77 | 7 s |
| 14 | 708 | 28426 | 3055386 | 135+101 | 12 s |
| 15 | 1039 | 49201 | 6987430 | 176+135 | 28 s |
Files per m: A_mM.npz (7/17/44 MB), sdp_mM.npz (0.1-0.4 MB), setup_mM.pkl (6/15/35 MB), prep_mM.json. Numeric rank = rows for all (full row rank). Total 121 MB.

## Sanity
- torch float64 4096^2 matmul on GPU OK.
- flint fmpq_mat solve OK ([-4],[9/2]).
- sdpap: tiny SDP (min tr X s.t. X12=1) solved, pdOPT, obj 1.00000003 (script sdpap_test.py; note SymCone s must be a tuple, J=SymCone(f=#eq)).
- `python3 -O tracks/D_order14/verify_strict.py <abs cert> <abs out>` on certs/exact_sos_m6.json: exact_identity, positive_definite_grams, ibp_identities_verified all true, 0.03 s. Output verify_m6_scratch.json. CLI takes explicit output path (no overwrite of certs/). It reads src/verify_sos_standalone.py from ~/gcm-logconcave-counterexamples (not the repo copy) and chdirs there, so pass absolute paths.
## Failures
None (first sdpap attempt failed only from my wrong SymCone usage).
