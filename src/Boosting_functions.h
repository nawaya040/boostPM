#ifndef Boosting_FUNCTIONS
#define Boosting_FUNCTIONS

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;



//Make the transition matrix
arma::mat CreateXi_Boosting(const int& I, const List& model_parameters_list, const int& k){
  arma::mat xi(1,1);
  xi(0,0) = 1;
  return xi;
}

//The hyper-parameter of the learing rate
  double c_g;

void pre_compute_Boosting(const int& G, const int& I, const int& NL, const List& model_parameters_list){
  //The hyper-parameter of the learing rate
  c_g = model_parameters_list["c"];
}

//Compute the marginal likelihood
double log_ML_compute_Boosting(const int& G, const int& I, const arma::ivec& n_l_vec,
                      const arma::ivec& n_r_vec,const double& L_input, const int& NL, const int& V, const int& k){
  int n_l = n_l_vec(0);
  int n_r = n_r_vec(0);
  int n = n_l + n_r;

  double out;

  if(n == 0){
    out = 0;
  }else{
    double prec = (1-c_g)/c_g * ((double) n);

    double alpha_l = prec * L_input;
    double alpha_r = prec * (1-L_input);

    out = lgamma(alpha_l+n_l) + lgamma(alpha_r+n_r) - lgamma(alpha_l+n_l+alpha_r+n_r) -
            (lgamma(alpha_l) + lgamma(alpha_r) - lgamma(alpha_l + alpha_r));
  }

  return out;
}


//Compute the posterior mean
double PostMean_Boosting(const int& I, const ivec& n_l_vec, const ivec& n_r_vec, const double& L_input, const int& V, const int& g, const int& k){
  int n_l = n_l_vec(0);
  int n_r = n_r_vec(0);
  int n = n_l + n_r;

  double out;

  if(n == 0){
    out = L_input;
  }else{
    double prec = (1-c_g)/c_g * ((double) n);

    double alpha_l = prec * L_input;
    double alpha_r = prec * (1-L_input);

    out = (alpha_l+n_l) / (alpha_l+n_l + alpha_r+n_r);
  }

  return out;
}

#endif
