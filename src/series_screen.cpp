// Numerical screening only. Every reported hit must pass independent Arb bounds.
#include <cmath>
extern "C" double evaluate_series(int m,int count,const int* ps,const int* qs,const int* ks,const double* weights,double a,double b,double kappa) {
 double ap[129],bp[129],filter[33],tp[1025];ap[0]=bp[0]=1;
 for(int i=1;i<=128;i++){ap[i]=ap[i-1]*a;bp[i]=bp[i-1]*b;}
 for(int k=0;k<=32;k++)filter[k]=exp(-k*k/(2*kappa));
 for(int B=-512;B<=512;B++){
  double t=1,c=1;
  for(int j=1;j<m;j++){c*=double(m-j+1)*(m-j)/j*kappa;t=t*B+c;}
  tp[B+512]=t*B;
 }
 double periodic=0,correction=0;
 for(int i=0;i<count;i++){
  int p=ps[i],q=qs[i],k=ks[i],B=(p+4*q-k*k)/2;
  double v=weights[i]*ap[p]*bp[q]*filter[k]*tp[B+512];
  double y=v-correction,next=periodic+y;correction=(next-periodic)-y;periodic=next;
 }
 double factorial=tgamma(m),base=.5*factorial*pow(kappa,m);
 double xi=1+a*filter[1]+b*filter[2];
 double moment=m*factorial*pow(kappa,m-1)*(a/2*filter[1]+2*b*filter[2]);
 return (base+(periodic+moment)/xi)/(factorial*pow(kappa,m));
}
