# Verified environment

Read-only readiness checks passed during the active overnight campaign at 2026-11-17T07:05:01.467496+00:00. GitHub CLI authentication, network access, repository origin, NVIDIA access, formula checks, four-family GPU validation, and Arb certification prerequisites all passed.

The visible GPU is an NVIDIA RTX A6000 with 49,140 MiB VRAM. The CPU environment exposes 8 logical CPUs and approximately 62 GiB RAM, rather than the requested 6/48 specification; CPU pools are limited by the research configuration and memory use is monitored against the 16 GB budget. Current cgroup memory usage, including file cache, remains below 16 GB. No guarantee of future uptime is implied.

See `certs/preflight.json`, `certs/cuda_pipeline_test.json`, `certs/cuda_higher_tests.json`, and live `checkpoints/` state for evidence. Earlier failed checks were superseded after the environment was fixed.
