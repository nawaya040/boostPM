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
         int J,
         double margin_size,
         double eta_subsample,
         int max_n_var
          ){
  
  int d = X.n_cols;
  mat support_store = zeros(d,2);
  vec log_width_store = zeros(d);
  
  if(margin_size !=0.0){
    //Rescale the observation points based on the minimum and maximum values
    for(int j=0; j<d; j++){
      double min_j = min(X.col(j));
      double max_j = max(X.col(j));
      double width_j = max_j - min_j;
      
      double m_resize = min(X.col(j)) - margin_size * width_j;
      double M_resize = max(X.col(j)) + margin_size * width_j;

      support_store(j,0) = m_resize;
      support_store(j,1) = M_resize;
           
      X.col(j) = (X.col(j) - m_resize) / (M_resize - m_resize);
      
      log_width_store(j) = log(M_resize - m_resize);
    }
  }else{
    for(int j=0; j<d; j++){
      support_store(j,0) = 0;
      support_store(j,1) = 1;
    }
  }
  
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
                              J,
                              eta_subsample,
                              max_n_var
                              );
  
  List out;
  
  //Run the boosting algorithm 
  my_boosting.boosting();

  out = my_boosting.output(support_store);
  
  //output the result
  return out;
}
