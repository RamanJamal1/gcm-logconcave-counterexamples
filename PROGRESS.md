# GCM research progress

Status: **PAUSED — workflow preflight only**, per user instruction.

The research uses Boltzmann entropy H = integral p log p and heat generator
(1/2) d²/dx², matching Zou et al. (2026).

- Literature verified through web retrieval; direct terminal downloads currently fail DNS.
- Exact reduced symbolic integrands generated through order 8; Gaussian controls pass.
- Independent 100-digit asymmetric heat-convolution checks pass: relative errors
  1.11e-22, 1.92e-22, and 1.16e-19 at orders 6, 7, and 8.
- Five-case CPU smoke test passes; 10 evaluations, maximum grid discrepancy 3.27e-15.
- No sustained search started; no counterexample or minimal-order theorem claimed.
- GitHub CLI is installed; API connectivity and GPU access have not passed preflight.
- Local git initialization and checkpoint commit pass. Remote creation failed at API connectivity.

See `PREFLIGHT_REPORT.md` and `certs/` for verification results. The requested
overnight run must not be treated as operational until these checks pass.
