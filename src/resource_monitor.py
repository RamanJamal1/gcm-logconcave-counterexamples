"""Record aggregate resource observations through the overnight deadline."""
import json,time,datetime,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sample():
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()};r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_used_bytes_upper_bound':memory['MemTotal']-memory['MemAvailable'],'host_total_bytes':memory['MemTotal'],'disk_free_bytes':shutil.disk_usage(ROOT).free}
 try:
  p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8);util,used,total=map(int,p.stdout.strip().split(','));r.update(gpu_utilization_percent=util,gpu_memory_used_mib=used,gpu_memory_total_mib=total)
 except Exception as e:r['gpu_query_error']=str(e)
 return r
if __name__=='__main__':
 schedule=json.loads((ROOT/'checkpoints/overnight_schedule.json').read_text());deadline=datetime.datetime.fromisoformat(schedule['deadline_utc']).timestamp();state={'sampling_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'samples':0,'max_host_used_bytes_upper_bound':0,'max_gpu_used_mib':0,'host_budget_exceeded_observed':False}
 while time.time()<deadline:
  r=sample();state['samples']+=1;state['last']=r;state['max_host_used_bytes_upper_bound']=max(state['max_host_used_bytes_upper_bound'],r['host_used_bytes_upper_bound']);state['max_gpu_used_mib']=max(state['max_gpu_used_mib'],r.get('gpu_memory_used_mib',0));state['host_budget_exceeded_observed']|=r['host_used_bytes_upper_bound']>=16*1024**3
  (ROOT/'checkpoints/resource_observations.json').write_text(json.dumps(state,indent=2)+'\n');time.sleep(30)
 state['status']='deadline_reached';(ROOT/'checkpoints/resource_observations.json').write_text(json.dumps(state,indent=2)+'\n')
