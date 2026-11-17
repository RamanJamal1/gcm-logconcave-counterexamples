"""Build isolated upstream SDPA-DD and add a 30-digit sparse restart export."""
from pathlib import Path
import subprocess, re, os
root=Path(__file__).resolve().parents[1]; vendor=root/'.venv/vendor'; d=vendor/'sdpa-dd'
p=d/'sdpa_main.cpp'; s=p.read_text()
marker='    // com.display(fpOut);'
export=r'''
    // Local research patch: preserve all DD digits in SDPA sparse restart.
    if (const char* restart = getenv("SDPA_RESTART")) {
        std::ofstream out(restart); out.precision(30); out << std::scientific << " ";
        for(int j=0;j<m;j++) out << -currentPt.yVec.ele[j] << " ";
        out << "\n";
        for(int k=1;k<=2;k++) {
            DenseLinearSpace &q = k==1 ? currentPt.zMat : currentPt.xMat;
            for(int l=0;l<q.SDP_nBlock;l++) {
                int n=q.SDP_block[l].nRow;
                for(int i=0;i<n;i++) for(int j=i;j<n;j++)
                    out << k << " " << l+1 << " " << i+1 << " " << j+1 << " " << q.SDP_block[l].de_ele[i+n*j] << "\n";
            }
        }
    }
'''
if 'Local research patch:' not in s:p.write_text(s.replace(marker,export+marker))
make=(d/'Makefile.am').read_text().split('else\nsdpa_dd_SOURCES =',1)[1].split('endif',1)[0]
sources=re.findall(r'[\w/]+\.cpp',make)
cmd=['g++','-O2','-fopenmp','-pthread','-DUseMETIS=0','-I.','-Implapack',f'-I{vendor}/sdpa-multiprecision/spooles/build',f'-I{vendor}/qd/usr/include',*sources,f'{vendor}/sdpa-multiprecision/spooles/build/libspooles.a',f'-L{vendor}/qd/usr/lib/x86_64-linux-gnu',f'-Wl,-rpath,{vendor}/qd/usr/lib/x86_64-linux-gnu','-lqd','-o',str(root/'.venv/bin/sdpa_dd')]
print('Building',len(sources),'translation units',flush=True)
subprocess.run(cmd,cwd=d,check=True)
