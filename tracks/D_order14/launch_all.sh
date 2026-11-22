#!/bin/bash
T=~/gcm-logconcave-counterexamples/tracks/D_order14
echo "$(date -u +%FT%TZ) launching 4 parallel chains: m14/128 m14/256 m15/128 m15/256" >> $T/logs/pipeline.log
for job in "14 128" "14 256" "15 128" "15 256"; do setsid nohup $T/chain.sh $job > /dev/null 2>&1 < /dev/null & done
