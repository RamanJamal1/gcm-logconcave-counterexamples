# Track G — exact pipeline (WIP, recovered)

Recovered local copies of Engineer 2's files after the session ended; they were never run end-to-end in a committed state.
Intended pipeline: exact_project.py (FLINT exact minimum-norm projection of a high-precision seed) -> pd_check.py (fast rigorous PD) ->
precheck.py (fast exact identity check) -> make_cert.py (certificate in reconstruct_quotient_fast format) -> run_pipeline.py (driver).
Final verdict must come from tracks/D_order14/verify_strict.py under python -O. No certificate produced.
