#!/bin/bash
# Unattended order-14 then order-15 exact-SOS pipeline. Survives logout (start with setsid nohup).
# Steps per order: SDPA-GMP solve -> numerical projection -> exact rational reconstruction ->
# strict exact verification (asserts rewritten, python -O) -> git commit on research branch (no push).
R=~/gcm-logconcave-counterexamples; T=$R/tracks/D_order14; L=$T/logs/pipeline.log; cd $R
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L; echo "{\"time\":\"$(date -u +%FT%TZ)\",\"state\":\"$*\"}" > $T/data/status.json; }
log "pipeline start pid $$"
while pgrep -f "gmp_solve.py 14 128 10 3" >/dev/null; do sleep 30; done; log "timing test finished: $(tail -1 $T/logs/gmp_test.log | cut -c1-300)"
for m in 14 15; do
  for bits in 128 256; do
    tag=m${m}_${bits}b_it200; log "order $m: SDPA-GMP solve bits=$bits"
    python3 $T/gmp_solve.py $m $bits 10 200 > $T/logs/gmp_$tag.log 2>&1 || { log "order $m solve bits=$bits crashed: $(tail -2 $T/logs/gmp_$tag.log | tr '\n' ' ')"; continue; }
    log "order $m solve done: $(tail -1 $T/logs/gmp_$tag.log | cut -c1-400)"
    python3 $T/proj.py $m $T/data/seed_$tag.json $T/data/seedproj_$tag.json > $T/logs/proj_$tag.log 2>&1 || { log "order $m projection failed: $(tail -2 $T/logs/proj_$tag.log | tr '\n' ' ')"; continue; }
    log "order $m projected: $(tail -1 $T/logs/proj_$tag.log)"
    ok=0
    for src in seedproj seed; do
      python3 src/reconstruct_quotient_fast.py $m $T/data/${src}_$tag.json > $T/logs/recon_${src}_$tag.log 2>&1 || { log "order $m exact reconstruction from $src failed: $(tail -1 $T/logs/recon_${src}_$tag.log | cut -c1-300)"; continue; }
      cp certs/exact_fast_quotient_m$m.json $T/data/exact_quotient_m${m}_from_${src}_$tag.json
      if python3 -O $T/verify_strict.py certs/exact_fast_quotient_m$m.json $T/data/verify_m${m}.json > $T/logs/verify_$tag.log 2>&1; then
        log "ORDER $m VERIFIED (strict, python -O): $(cut -c1-300 $T/logs/verify_$tag.log)"; ok=1
        git add certs/exact_fast_quotient_m$m.json tracks/D_order14/*.py tracks/D_order14/pipeline.sh tracks/D_order14/data/verify_m$m.json tracks/D_order14/data/status.json tracks/D_order14/logs >> $L 2>&1
        git commit -q -m "Order $m exact SOS certificate: verified by strict standalone verifier under python -O

S_$m >= 0 for log-concave densities under the repository's stated analytic hypotheses
(same certificate form and verifier as orders 6-11, 13). Seed: SDPA-GMP $bits-bit, $src.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >> $L 2>&1 && log "order $m committed $(git log --oneline -1)"
        break
      else
        rm -f certs/exact_fast_quotient_m$m.json; log "order $m verification FAILED from $src: $(tail -2 $T/logs/verify_$tag.log | tr '\n' ' ' | cut -c1-300)"
      fi
    done
    [ $ok = 1 ] && break
  done
  [ $ok = 1 ] || log "order $m: no verified certificate after all attempts"
done
log "pipeline end"
