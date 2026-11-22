#!/bin/bash
# One unattended chain: SDPA-GMP solve (order m, given bits) -> projection -> exact reconstruction
# -> strict verification (asserts rewritten, python -O) -> commit on research branch. No push.
m=$1; bits=$2; R=~/gcm-logconcave-counterexamples; T=$R/tracks/D_order14; L=$T/logs/pipeline.log; cd $R
tag=m${m}_${bits}b_it200
log(){ echo "$(date -u +%FT%TZ) [m$m/${bits}b] $*" >> $L; }
log "solve start"
python3 $T/gmp_solve.py $m $bits 1 200 > $T/logs/gmp_$tag.log 2>&1 || { log "solve crashed: $(tail -2 $T/logs/gmp_$tag.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
log "solve done: $(tail -1 $T/logs/gmp_$tag.log | cut -c1-350)"
[ -f $T/data/VERIFIED_m$m ] && { log "order already verified by another chain; stop"; exit 0; }
python3 $T/proj.py $m $T/data/seed_$tag.json $T/data/seedproj_$tag.json > $T/logs/proj_$tag.log 2>&1 && log "projected: $(tail -1 $T/logs/proj_$tag.log)" || log "projection failed: $(tail -1 $T/logs/proj_$tag.log | cut -c1-200)"
exec 9>$T/data/lock_m$m; flock 9
[ -f $T/data/VERIFIED_m$m ] && { log "order already verified; stop"; exit 0; }
for src in seedproj seed; do
  [ -f $T/data/${src}_$tag.json ] || continue
  python3 src/reconstruct_quotient_fast.py $m $T/data/${src}_$tag.json > $T/logs/recon_${src}_$tag.log 2>&1 || { log "exact reconstruction from $src failed: $(tail -1 $T/logs/recon_${src}_$tag.log | cut -c1-250)"; continue; }
  cp certs/exact_fast_quotient_m$m.json $T/data/exact_quotient_m${m}_from_${src}_$tag.json
  if python3 -O $T/verify_strict.py certs/exact_fast_quotient_m$m.json $T/data/verify_m${m}.json > $T/logs/verify_${src}_$tag.log 2>&1; then
    touch $T/data/VERIFIED_m$m; log "ORDER $m VERIFIED (strict, python -O) from $src: $(cut -c1-250 $T/logs/verify_${src}_$tag.log)"
    for i in 1 2 3 4 5; do
      git add certs/exact_fast_quotient_m$m.json tracks/D_order14/*.py tracks/D_order14/*.sh tracks/D_order14/data/verify_m$m.json tracks/D_order14/data/exact_quotient_m${m}_from_${src}_$tag.json tracks/D_order14/logs >> $L 2>&1 && \
      git commit -q -m "Order $m exact SOS certificate verified (strict standalone verifier, python -O)

S_$m >= 0 for log-concave densities under the repository's stated analytic hypotheses
(same certificate form and verifier as orders 6-11 and 13). Seed: SDPA-GMP ${bits}-bit ($src).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> $L 2>&1 && { log "committed $(git log --oneline -1)"; break; }
      sleep 20
    done
    exit 0
  else
    rm -f certs/exact_fast_quotient_m$m.json; log "verification FAILED from $src: $(tail -1 $T/logs/verify_${src}_$tag.log | cut -c1-250)"
  fi
done
log "chain ended without certificate"
