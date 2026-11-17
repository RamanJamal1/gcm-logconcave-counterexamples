"""Run rationalization after the saved B completion; never infer completion from time."""
import os,json,time,datetime,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 lock=open(ROOT/'.git/campaign_c.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.chdir(ROOT)
 while True:
  p=ROOT/'checkpoints/campaign_B.json'
  if p.exists():
   try:state=json.loads(p.read_text())
   except json.JSONDecodeError:state={}
   if state.get('status')=='bounded_run_complete':break
  time.sleep(30)
 schedule=json.loads((ROOT/'checkpoints/overnight_schedule.json').read_text());schedule.update(phase='C',status='campaign_C_running',campaign_C_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(ROOT/'checkpoints/overnight_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 from campaign_c import main as campaign
 campaign(seconds=3600,batch=65536,sync=True)
 schedule.update(phase='certification_and_report',status='campaign_C_complete');(ROOT/'checkpoints/overnight_schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
 print('Campaign C complete',flush=True)
if __name__=='__main__':main()
