"""Run remaining B segments, retaining checkpoints after interrupted workers."""
import os,json,time,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if __name__=='__main__':
 while True:
  state=json.loads((ROOT/'checkpoints/campaign_B.json').read_text())
  if state.get('status')=='bounded_run_complete':break
  if state.get('status')=='running':
   # Another explicitly launched segment owns the GPU until it checkpoints.
   time.sleep(20);continue
  with open(ROOT/'logs/campaign_B_chunks.log','a') as log:
   result=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','src/run_campaign_b_chunk.py'],stdout=log,stderr=subprocess.STDOUT)
  print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'segment exit',result.returncode,flush=True)
  if result.returncode:
   state=json.loads((ROOT/'checkpoints/campaign_B.json').read_text());state['status']='interrupted';state.setdefault('errors',[]).append(f'Chunk process exit {result.returncode}');(ROOT/'checkpoints/campaign_B.json').write_text(json.dumps(state,indent=2)+'\n');time.sleep(5)
 print('B all active segments complete',flush=True)
