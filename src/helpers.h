#ifndef HELPERS
#define HELPERS

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

ivec Count_groups(const int& G, const ivec& groups);
int get_position_index(const double& x, const double& dleft, const double& dright, const int& NL);
vec colsum_for_cube(const cube& Q, const int& ncol);
ivec colsum_for_icube(const icube& Q, const int& ncol);
double ColSum_log(const arma::vec& vx);
double Sum_log(const arma::mat& mx);
arma::vec Normalize_log(const arma::vec& vx);
int OneSample(const arma::vec& vw);
double ComputeESS(const arma::vec& vw);
arma::uvec Resample_index(const int length_output,const arma::colvec& vw);
void Resample_cube(const int length_output, arma::cube& cx, const arma::uvec& indices);
void Resample_icube(const int length_output, arma::icube& cx, const arma::uvec& indices);
void Resample_ucube(const int length_output, arma::ucube& cx, const arma::uvec& indices);


arma::vec logBeta(const arma::vec& va,const arma::vec& vb);

#endif