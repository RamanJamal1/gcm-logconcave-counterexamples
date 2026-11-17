"""Continue 25-minute Git backup coverage during the final proof phase."""
import datetime,time,json,subprocess,fcntl
from pathlib import Path
from checkpoint import checkpoint
from checkpoint_io import write_json
ROOT=Path(__file__).resolve().parents[1]
def main():
 lock=open(ROOT/'.git/sync-watchdog.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);schedule=json.loads((ROOT/'checkpoints/overnight_schedule.json').read_text());end=datetime.datetime.fromisoformat(schedule['deadline_utc']).timestamp()+3600
 while time.time()<end and not (ROOT/'certs/github_issue.json').exists():
  r=subprocess.run(['git','log','-1','--format=%ct'],cwd=ROOT,text=True,capture_output=True,timeout=20);last=float(r.stdout.strip()) if r.returncode==0 else 0
  if time.time()-last>=1500:
   result=checkpoint('periodic proof-phase backup');write_json(ROOT/'checkpoints/sync_watchdog_last_push.json',result)
  write_json(ROOT/'checkpoints/sync_watchdog.json',{'status':'running','heartbeat_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interval_seconds':1500,'stop_on_issue_receipt':True})
  time.sleep(30)
 write_json(ROOT/'checkpoints/sync_watchdog.json',{'status':'stopped','reason':'issue posted or final grace interval ended','stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':main()
