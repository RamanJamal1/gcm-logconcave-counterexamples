"""Visible heartbeat and exit evidence for long research subprocesses.

Signals are recorded and forwarded, never ignored. This wrapper does not
restart a deliberately terminated process or claim that silence means success.
"""
import argparse,subprocess,time,signal,json,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command[1:] if a.command and a.command[0]=='--' else a.command
path=ROOT/f'checkpoints/process_{a.name}.json';start=time.time();state={'name':a.name,'command':command,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'starting'};child=None

def save():path.write_text(json.dumps(state,indent=2)+'\n')
def stop(signum,frame):
 state.update(status='signal_received',signal=signum,signal_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start);save()
 if child is not None and child.poll() is None:child.send_signal(signum)
 raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
with open(ROOT/a.log,'a',buffering=1) as log:
 child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='running',child_pid_in_namespace=child.pid);save()
 while child.poll() is None:
  state.update(heartbeat_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start);save();print(f"{a.name}: running, elapsed {state['elapsed_seconds']:.0f}s, log {a.log}",flush=True);time.sleep(30)
 state.update(status='exited',returncode=child.returncode,elapsed_seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();print(json.dumps(state),flush=True)
 raise SystemExit(child.returncode)
