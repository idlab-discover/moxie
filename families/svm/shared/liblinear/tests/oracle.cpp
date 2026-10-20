// Validation adapter only. The unmodified pinned sklearn sources are linked.
#include "linear.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
static double dot(int n,const double*x,int sx,const double*y,int sy) { double v=0; for(int i=0;i<n;++i)v+=x[i*sx]*y[i*sy];return v; }
static void axpy(int n,double a,const double*x,int sx,double*y,int sy) {for(int i=0;i<n;++i)y[i*sy]+=a*x[i*sx];}
static void scal(int n,double a,const double*x,int sx) {for(int i=0;i<n;++i)const_cast<double*>(x)[i*sx]*=a;}
static double norm(int n,const double*x,int sx) {return std::sqrt(dot(n,x,sx,x,sx));}
static void quiet(const char*) {}
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    std::ifstream f(argv[1]);
    int n,p,k,solver,intercept,limit; unsigned seed;
    double c,epsilon,tol,scale;
    f>>n>>p>>k>>solver>>c>>epsilon>>tol>>intercept>>scale>>limit>>seed;
    int q=p+intercept;
    std::vector<std::vector<feature_node>> storage(n);
    std::vector<feature_node*> rows(n);
    for(int i=0;i<n;++i){
        for(int j=0;j<p;++j){double v; f>>v; if(v!=0)storage[i].push_back({j+1,v});}
        if(intercept)storage[i].push_back({p+1,scale});
        storage[i].push_back({-1,0});rows[i]=storage[i].data();
    }
    std::vector<double> y(n),sw(n),cw(k);std::vector<int> labels(k);
    for(auto&v:y)f>>v;for(auto&v:sw)f>>v;for(auto&v:cw)f>>v;
    for(int i=0;i<k;++i)labels[i]=i;
    if(!f)throw std::runtime_error("invalid fixture");
    problem prob{n,q,y.data(),rows.data(),intercept?scale:-1,sw.data()};
    parameter param{solver,tol,c,k,labels.data(),cw.data(),limit,epsilon};
    BlasFunctions blas{dot,axpy,scal,norm};
    set_seed(seed);set_print_string_function(quiet);
    if(auto error=check_parameter(&prob,&param))throw std::runtime_error(error);
    auto model=train(&prob,&param,&blas);
    int r=(solver==4)?model->nr_class:((model->nr_class==2)?1:model->nr_class);
    int t=(solver==4)?1:r;
    std::cout<<r<<' '<<p<<' '<<0<<'\n'<<std::setprecision(17);
    for(int i=0;i<r;++i)for(int j=0;j<p;++j)std::cout<<model->w[j*r+i]<<'\n';
    for(int i=0;i<r;++i)std::cout<<(intercept?model->w[p*r+i]*scale:0)<<'\n';
    for(int i=0;i<t;++i)std::cout<<model->n_iter[i]<<'\n';
    free_and_destroy_model(&model);
}
