// Double-precision numerical seed locator; no result is a proof.
#include <sdpa_call.h>

#include <cstdlib>
#include <cstdio>
int main(int argc,char**argv) {
 if(argc<5){fprintf(stderr,"usage: segment input output iterations bits [initial]\n");return 2;}
 
 FILE* sink=fopen("/dev/null","w");SDPA p;p.setNumThreads(1);p.setDisplay(stdout);p.setResultFile(sink);p.setParameterMaxIteration(atoi(argv[3]));p.setParameterEpsilonStar(1e-10);p.setParameterEpsilonDash(1e-10);p.setParameterLambdaStar(1000);p.setParameterUpperBound(1e14);p.setParameterLowerBound(-1e14);
 p.readInput(argv[1],sink,SDPA::SPARSE);
 if(argc>5){p.setInitPoint(true);p.readInit(argv[5],sink,SDPA::SPARSE);}
 p.initializeSolve();p.solve();
 FILE*f=fopen(argv[2],"w");if(!f)return 3;
 fprintf(f," ");auto x=p.getResultXVec();for(int j=0;j<p.getConstraintNumber();j++)fprintf(f,"%.17e ",x[j]);fprintf(f,"\n");
 for(int k=1;k<=2;k++)for(int l=1;l<=p.getBlockNumber();l++){
  int n=p.getBlockSize(l);auto q=k==1?p.getResultXMat(l):p.getResultYMat(l);
  for(int i=0;i<n;i++)for(int j=i;j<n;j++){fprintf(f,"%d %d %d %d ",k,l,i+1,j+1);fprintf(f,"%.17e\n",q[i+n*j]);}
 }
 fclose(f);char phase[100];p.getPhaseString(phase);printf("SEGMENT phase=%s iterations=%d primal_error=%.12g dual_error=%.12g\n",phase,p.getIteration(),p.getPrimalError(),p.getDualError());fflush(stdout);p.terminate();return 0;
}
