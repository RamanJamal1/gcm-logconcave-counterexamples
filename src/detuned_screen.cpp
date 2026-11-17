#include <cmath>
extern "C" double evaluate_detuned(int m,int count,const int* ps,const int* qs,const int* ls,const int* rs,const double* weights,double a,double b,double kappa,double omega) {
 double ap[129],bp[129],co[64];ap[0]=bp[0]=1;co[0]=1;
 for(int i=1;i<=128;i++){ap[i]=ap[i-1]*a;bp[i]=bp[i-1]*b;}
 for(int j=1;j<m;j++)co[j]=co[j-1]*double(m-j+1)*(m-j)/j*kappa;
 double sum=0,correction=0;
 for(int i=0;i<count;i++){
  double frequency=ls[i]+omega*rs[i];if(fabs(frequency)>6)continue;
  double B=(ps[i]+qs[i]*omega*omega-frequency*frequency)/2;
  double T=1;for(int j=1;j<m;j++)T=T*B+co[j];T*=B;
  double value=weights[i]*ap[ps[i]]*bp[qs[i]]*T*exp(-frequency*frequency/(2*kappa));
  double y=value-correction,next=sum+y;correction=(next-sum)-y;sum=next;
 }
 double f=tgamma(m),e1=exp(-1/(2*kappa)),e2=exp(-omega*omega/(2*kappa));
 double moment=m*f*pow(kappa,m-1)*(a/2*e1+b*omega*omega/2*e2);
 return .5*f*pow(kappa,m)+(sum+moment)/(1+a*e1+b*e2);
}
