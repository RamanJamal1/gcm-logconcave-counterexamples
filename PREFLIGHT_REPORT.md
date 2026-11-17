# Workflow preflight

**NOT READY for unattended overnight execution. Research remains paused.**

The CLI is now installed (`gh 2.4.0`), and `gh auth setup-git` succeeds.
However, `gh api user` and the requested private-repository creation command
both fail with `error connecting to api.github.com`. A direct HTTPS probe
reports `Could not resolve host: api.github.com`. Although `gh auth status`
labels the token invalid, network failure prevents determining whether the
token itself is valid. No token was printed or committed.

## Passed

- Local repository initialized on `main`; source, JSON checkpoints, formulas,
  and progress tracked. A local checkpoint commit succeeded.
- `.gitignore` excludes virtual environments, Python caches, secrets, and raw
  binary dumps. The checkpoint utility also rejects staged files over 10 MiB.
- Exact symbolic entropy formulas and Gaussian controls through order 8.
- Independent asymmetric heat-convolution finite differences at 100 digits:
  relative errors 1.11e-22, 1.92e-22, and 1.16e-19 for orders 6, 7, and 8.
  Quadrature convergence also meets the requested 1e-11 threshold.
- Five controlled CPU cases across four density families, 10 evaluations,
  maximum 1,025-vs-4,097-grid relative discrepancy 3.27e-15.
- Found and repaired an unstable RBF mode calculation using bracketed Brent
  iteration. Interrupted pilot artifacts are marked invalid for conclusions.

## Failed or incomplete

- **GitHub:** remote create, push, and issue posting cannot be tested until
  this terminal can reach the API. No remote repository or issue was created.
- **GPU (updated verification):** the machine's Thunder configuration explicitly
  specifies `gpuType: a6000` and `gpuCount: 1`. CUDA/NVML are routed through
  Thunder's globally preloaded library. The GPU service socket is rejected with
  `PermissionError: [Errno 1] Operation not permitted` by this command sandbox.
  Thus the A6000 allocation is configured, but driver access and CUDA execution
  remain unverified. Missing local NVIDIA device nodes do not imply no GPU.
  See `certs/gpu_access_probe.json` for the redacted hardware-access evidence.
- **Hardware mismatch:** visible CPU/RAM are 8 CPUs and 62 GiB; disk is 193 GiB.
  These observations differ from the requested 6 CPUs / 48 GB specification.
- **Dependencies:** JAX, optax, evosax, and python-flint are absent; terminal
  package-network access fails. NumPy, SciPy, SymPy, mpmath, and Torch exist.
- **Implementation:** validated CPU evaluator exists; CUDA batching,
  mixed-precision optimization, resume, automatic certification, and the
  timed/new-elite GitHub sync hooks are not implemented or validated.
- **Certification:** numerical validation is not an interval proof. No
  certified counterexample or minimal-order result exists from this work.

## Recheck

Run `python src/preflight.py` from this repository. It records machine-readable
results in `certs/preflight.json` and exits nonzero while gates remain blocked.
It does not launch the search or mutate GitHub. The local checkpoint utility is
`python src/checkpoint.py --local-only --reason 'preflight'`.

Before launch, the terminal needs working GitHub/package access and GPU backend
access, followed by implementation and end-to-end testing of the remaining
pipeline. No unattended loop is running.
