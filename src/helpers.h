#ifndef HELPERS_H
#define HELPERS_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

int OneSample(const arma::vec& vw);
int OneSample_uniform(const int size);
double log_beta(const double a, const double b);
double log_sum_vec(const vec& log_x);
double log_sum_mat(const mat& log_x);
vec log_normalize_vec(const vec& log_x);
mat log_normalize_mat(const mat& log_x);
double second_max(vec x);
double second_min(vec x);

#endif