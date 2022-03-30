// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

int OneSample(const arma::vec& vw){
  double u = R::runif(0,1);
  arma::uvec out = find(cumsum(vw) > u, 1);
  
  return (int) out(0);
}

int OneSample_uniform(const int size){
  vec vw(size);
  vw.fill(1.0 / (double) size);
  return OneSample(vw);
}

double log_beta(const double a, const double b)
{
  return lgamma(a) + lgamma(b) - lgamma(a + b);
}

double log_sum_vec(const vec& log_x){
  double log_x_max = log_x.max();
  return log_x_max + log(sum(exp(log_x - log_x_max)));
}

double log_sum_mat(const mat& log_x){
  return log_sum_vec(vectorise(log_x));
}

vec log_normalize_vec(const vec& log_x){
  return exp(log_x - log_sum_vec(log_x));
}

mat log_normalize_mat(const mat& log_x){
  int n_rows = log_x.n_rows;
  int n_cols = log_x.n_cols;
  return reshape(log_normalize_vec(vectorise(log_x)), n_rows, n_cols);
}


double second_max(vec x){
  int len = x.n_rows;
  
  if(len < 2){
    stop("error: the input vector too small");
  }

  double first, second;
  
  if(x(0) > x(1)){
    first = x(0);
    second = x(1);
  }else{
    first = x(1);
    second = x(0);
  }
  
  for(int i=2; i<len; i++){
    double x_current = x(i);
    
    if(x_current > first){
      second = first;
      first = x_current;
    }else if(x_current > second){
      second = x_current;
    }
  }
  
  return second;
}

double second_min(vec x){
  return (-1.0) * second_max((-1.0) * x);
}




