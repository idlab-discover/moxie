// Validation only: compile the unmodified, pinned scikit-learn LIBSVM source.
#define _DENSE_REP
#include "svm.cpp"
#include <iomanip>
#include <iostream>
#include <vector>

static double dot(int n, const double* x, int sx, const double* y, int sy) {
    double v = 0;
    for (int i=0; i<n; ++i) v += x[i*sx]*y[i*sy];
    return v;
}
static void quiet(const char*) {}
static void read_double(double& value) { std::string token; std::cin >> token; value = std::stod(token); }
template<class T> static void output(const char* name, const T* values, int n) {
    std::cout << name;
    for (int i=0;i<n;++i) std::cout << ' ' << values[i];
    std::cout << '\n';
}
int main(int argc, char** argv) {
    if (argc == 1) {
        int n,p,m,k; svm_parameter param{};
        std::cin>>n>>p>>m>>k>>param.svm_type>>param.kernel_type>>param.degree;
        read_double(param.gamma);read_double(param.coef0);read_double(param.cache_size);read_double(param.eps);read_double(param.C);read_double(param.nu);read_double(param.p);
        std::cin>>param.shrinking>>param.max_iter>>param.probability>>param.random_seed;
        std::vector<double> x(n*p), y(n), w(n), cw(k), query(m*p);
        for(auto& v:x)read_double(v);for(auto& v:y)read_double(v);for(auto& v:w)read_double(v);
        for(auto& v:cw)read_double(v);for(auto& v:query)read_double(v);
        std::vector<int> labels(k);std::vector<svm_node> nodes(n);
        for(int i=0;i<n;++i)nodes[i]={p,i,x.data()+i*p};
        for(int i=0;i<k;++i)labels[i]=i;
        if(param.svm_type<2){param.nr_weight=k;param.weight_label=labels.data();param.weight=cw.data();}
        svm_problem problem{n,y.data(),nodes.data(),w.data()};BlasFunctions blas{dot};svm_set_print_string_function(quiet);
        if(const char* error=svm_check_parameter(&problem,&param)){std::cerr<<error;return 3;}
        int status=0;auto* model=svm_train(&problem,&param,&status,&blas);
        int q=param.svm_type<2?k*(k-1)/2:1, rows=param.svm_type<2?k-1:1;
        std::cout<<std::setprecision(17);output("raw_support",model->sv_ind,model->l);
        std::vector<int> original_support(model->l);for(int i=0;i<model->l;++i)original_support[i]=model->SV[i].ind;
        output("support",original_support.data(),model->l);
        if(param.svm_type<2)output("n_support",model->nSV,k);else output("n_support",&model->l,1);
        std::cout<<"dual";for(int i=0;i<rows;++i)for(int j=0;j<model->l;++j)std::cout<<' '<<model->sv_coef[i][j];std::cout<<'\n';
        output("rho",model->rho,q);output("iterations",model->n_iter,q);output("status",&status,1);
        output("prob_a",model->probA,model->probA?q:0);output("prob_b",model->probB,model->probB?q:0);
        std::vector<double> decisions(m*q), probabilities(m*k), predictions(m);
        for(int i=0;i<m;++i){svm_node node{p,i,query.data()+i*p};predictions[i]=svm_predict_values(model,&node,decisions.data()+i*q,&blas);
            if(param.svm_type<2&&param.probability)svm_predict_probability(model,&node,probabilities.data()+i*k,&blas);}
        output("decisions",decisions.data(),m*q);output("probabilities",probabilities.data(),param.svm_type<2&&param.probability?m*k:0);
        output("predictions",predictions.data(),m);svm_free_and_destroy_model(&model);return 0;
    }
    if (argc != 12) return 2;
    const int form=std::stoi(argv[1]), kind=std::stoi(argv[2]), n=std::stoi(argv[3]),
        p=std::stoi(argv[4]), k=std::stoi(argv[5]), weighted=std::stoi(argv[6]);
    std::vector<double> raw(n*p), x(n*(kind==4?n:p)), y(n), w(n), cw(k);
    std::vector<int> labels(k);
    std::vector<svm_node> nodes(n);
    for(int i=0;i<n;++i) {
        y[i]=form<2 ? i%k : (form==2 ? 1 : ((i*13)%23-11)/7.0);
        w[i]=weighted ? (i%7==0 ? (weighted==2 ? -0.5 : 0.0) : 0.5+(i%5)*0.4) : 1;
        for(int j=0;j<p;++j) raw[i*p+j]=((i*17+j*11)%37-18)/13.0 + (i%k)*2.0 + (form==2 ? 2.0 : 0.0);
    }
    for(int i=0;i<n;++i) {
        int width=kind==4?n:p;
        for(int j=0;j<width;++j) {
            if(kind!=4) x[i*width+j]=raw[i*p+j];
            else x[i*width+j]=dot(p,raw.data()+i*p,1,raw.data()+j*p,1);
        }
        nodes[i]={width,i,x.data()+i*width};
    }
    for(int i=0;i<k;++i) { labels[i]=i; cw[i]=weighted ? 0.7+0.4*i : 1; }
    svm_parameter param{};
    param.svm_type=form;param.kernel_type=kind;param.degree=3;param.gamma=0.4;param.coef0=0.2;
    param.cache_size=std::stod(argv[11]);param.eps=1e-7;param.C=1.7;param.nu=0.35;param.p=0.12;
    param.shrinking=std::stoi(argv[7]);param.max_iter=std::stoi(argv[8]);
    param.probability=std::stoi(argv[9]);param.random_seed=std::stoi(argv[10]);
    if(form<2) {param.nr_weight=k;param.weight_label=labels.data();param.weight=cw.data();}
    svm_problem problem{n,y.data(),nodes.data(),w.data()};
    BlasFunctions blas{dot};svm_set_print_string_function(quiet);
    if(const char* error=svm_check_parameter(&problem,&param)) {std::cerr<<error;return 3;}
    int status=0;
    svm_model* model=svm_train(&problem,&param,&status,&blas);
    const int q=form<2?k*(k-1)/2:1, rows=form<2?k-1:1;
    std::cout<<std::setprecision(17);
    output("raw_support",model->sv_ind,model->l);
        std::vector<int> original_support(model->l);for(int i=0;i<model->l;++i)original_support[i]=model->SV[i].ind;
        output("support",original_support.data(),model->l);
    if(form<2) output("n_support",model->nSV,k); else output("n_support",&model->l,1);
    std::cout<<"dual";for(int i=0;i<rows;++i)for(int j=0;j<model->l;++j)std::cout<<' '<<model->sv_coef[i][j];std::cout<<'\n';
    output("rho",model->rho,q);output("iterations",model->n_iter,q);output("status",&status,1);
    output("prob_a",model->probA,model->probA?q:0);output("prob_b",model->probB,model->probB?q:0);
    std::vector<double> decisions(n*q), probabilities(n*k);
    for(int i=0;i<n;++i) {
        svm_predict_values(model,&nodes[i],decisions.data()+i*q,&blas);
        if(form<2 && param.probability) svm_predict_probability(model,&nodes[i],probabilities.data()+i*k,&blas);
    }
    output("decisions",decisions.data(),n*q);
    output("probabilities",probabilities.data(),form<2&&param.probability?n*k:0);
    svm_free_and_destroy_model(&model);
}
