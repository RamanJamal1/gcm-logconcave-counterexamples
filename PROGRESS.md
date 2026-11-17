# GCM research progress

Updated: 2026-11-17T08:33:00.325963+00:00

Verified bracket: 12 ≤ m* ≤ 17. Exact minimality remains unresolved.

| Campaign | Saved status | FP32 candidates | FP64 refined candidates |
|---|---|---:|---:|
| campaign_A | bounded_run_complete | 11,468,800 | 154,112 |
| campaign_B | running | 65,536 | 16 |
| higher_campaign | stopped_for_new_order12_family | 1,630,208 | 25,472 |
| polynomial_campaign | interrupted_exit143 | 32,768 | 512 |
| polynomial_wide_campaign | running | 360,448 | 5,632 |

Recorded production totals: 13,557,760 FP32 candidate evaluations; 185,744 FP64 refinements. Warmups and readiness runs are excluded. Interrupted partial generations may not be counted.

Sharp proven normalized bounds: S6 Var^6 ≥ 60; S7 Var^7 ≥ 360; S8 Var^8 ≥ 2520.
Exact rational nonnegativity certificates cover orders 6–11.
The simple order-17 witness is independently certified by Fourier-series and spatial Arb calculations.
See MORNING_BRIEFING.md, checkpoints/research_status.json, and paper/draft_note.tex.
