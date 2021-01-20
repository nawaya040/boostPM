// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

ivec Count_groups(const int& G, const ivec& groups){
  ivec out(G);
  
  for(int g=0;g<G;g++){
    uvec indices_temp = find(groups == g);
    out(g) = indices_temp.n_rows;
  }

  return out;
} 

int get_position_index(const double& x, const double& dleft, const double& dright, const int& NL){
  double x_normalized = (x - dleft) / (dright - dleft);
  int out = floor(x_normalized * (double) NL);
  
  return out;
}

vec colsum_for_cube(const cube& Q, const int& ncol){
  vec out(ncol);
  
  for(int j=0;j<ncol;j++){
    out(j) = accu(Q.col(j));
  }
  
  return out;
} 

ivec colsum_for_icube(const icube& Q, const int& ncol){
  ivec out(ncol);
  
  for(int j=0;j<ncol;j++){
    out(j) = accu(Q.col(j));
  }
  
  return out;
} 


//Take a sum of a column vector
double ColSum_log(const arma::vec& vx)
{
  double max_temp = arma::max(vx);
  arma::vec vx_normalized = vx - max_temp;
  
  return max_temp + log(sum(exp(vx_normalized)));	
}

//Take a sum of an input matrix. The output is also a log value.
double Sum_log(const arma::mat& mx)
{
  double max_temp = mx.max();
  arma::mat mx_normalized = mx - max_temp;
  
  return max_temp + log(accu(exp(mx - max_temp)));	
}

//Normalize elements of a log-vector to make the sum 1.
arma::vec Normalize_log(const arma::vec& vx)
{
  int length = vx.n_rows;
  arma::vec out(length);
  
  arma::vec v_temp(length);
  double d_temp;
  
  for(int i=0;i<length; ++i)
  {
    v_temp = exp(vx - vx(i));
    d_temp = sum( exp(vx - vx(i)) );
    
    out(i) = pow(d_temp, -1);
  }
  
  return out;
}

//Sample from multinomial distribtion
//It outputs an index
int OneSample(const arma::vec& vw){
  double u = runif(1)(0);
  arma::uvec out = find(cumsum(vw) > u, 1);
  
  return (int) out(0);
}

//Computes the ESS
double ComputeESS(const arma::vec& vw){
  arma::vec vw2 = pow(vw,2.0);
  double deno = sum(vw2);
  double out = pow(deno,-1.0);
  
  return out;
}


//Output the indices for the resampling
//NOTICE::the indices are NOT sorted.
arma::uvec Resample_index(const int length_output,const arma::colvec& vw) {
  
  arma::colvec vu = runif(length_output);
  arma::colvec cum_w = cumsum(vw);
  
  arma::uvec indices = zeros<uvec>(length_output);
  
  for (int i = 0; i < length_output; i++) {
    arma::uvec indices_i = find(cum_w < vu(i));
    indices(i) = indices_i.n_elem;
  }
  
  return(sort(indices));
}

//Resample for cubes
void Resample_cube(const int length_output, arma::cube& cx, const arma::uvec& indices){
  arma::cube cx_temp = cx;
  for(int i=0; i<length_output ; i++){
    cx.row(i) = cx_temp.row(indices(i));
  }
}

//Resample for icubes
void Resample_icube(const int length_output, arma::icube& cx, const arma::uvec& indices){
  arma::icube cx_temp = cx;
  for(int i=0; i<length_output ; i++){
    cx.row(i) = cx_temp.row(indices(i));
  }
}

//Resample for ucubes
void Resample_ucube(const int length_output, arma::ucube& cx, const arma::uvec& indices){
  arma::ucube cx_temp = cx;
  for(int i=0; i<length_output ; i++){
    cx.row(i) = cx_temp.row(indices(i));
  }
}

//Compute the log of beta function
arma::vec logBeta(const arma::vec& va,const arma::vec& vb)
{
  return lgamma(va) + lgamma(vb) - lgamma(va + vb);
}
