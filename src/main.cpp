// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "class_boosting.h"
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List do_boosting(mat X,
         double precision,
         double alpha,
         double beta,
         double gamma,
         int max_resol,
         int num_each_dim,
         int num_second,
         double learn_rate,
         int min_obs,
         int nbins,
         double eta_subsample,
         double thresh_stop,
         int ntrees_wait,
         int max_n_var
          ){
  

  //Initialize the class object
  class_boosting my_boosting(X,
                              precision,
                              alpha,
                              beta,
                              gamma,
                              max_resol,
                              num_each_dim,
                              num_second,
                              learn_rate,
                              min_obs,
                              nbins,
                              eta_subsample,
                              thresh_stop,
                              ntrees_wait,
                              max_n_var
                              );
  
  List out;
  
  //Run the boosting algorithm 
  my_boosting.boosting();

  out = my_boosting.output();

  //output the result
  return out;
}
