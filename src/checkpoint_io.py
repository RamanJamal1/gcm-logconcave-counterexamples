"""Atomic file replacement for snapshots read concurrently by Git sync."""
import json,os,tempfile
from pathlib import Path
def write_text(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as f:
  f.write(text);f.flush();os.fsync(f.fileno());temporary=f.name
 os.replace(temporary,path)
def write_json(path,value):write_text(path,json.dumps(value,indent=2,allow_nan=False)+'\n')
