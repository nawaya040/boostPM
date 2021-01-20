// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "helpers.h"
#include "class_smc.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

// [[Rcpp::export]]
List SMCforPT(const mat X, //Data (n by d matrix)
              const int G, //Number of groups
              const ivec groups_input, //Group labels starting from 1
              const mat grid_points, //Matrix of grid points for density estimation
              const ivec groups_pred_input, //Group labels for grid points
              const double eta_R, //Tuning parameter for R
              const int I, //Number of states
              const List model_parameters_list, //Tuning parameters specific to the model
              const int M, //Number of particles
              const int max_K, //Maximum resolution
              const int NL, //# grid points + 1
              const double mixing_resample, //Parameter that controls the mixing of the resample steps
              const double thresh_resample, //Parameter that determines the threshold in the resampling steps
              const int minimum_size, //Partitioning is terminated if # observations is less than minimum_size
              const int do_density_est, //If=1, compute the predictive density at each grid point. Otherwise, the step is skipped
              const int method //1: APT, 2: MRS, 3: Other(customized)
){
        
 //Initialize the class object
 class_smc my_particles( X, 
                         G,
                         groups_input,
                         grid_points,
                         groups_pred_input,
                         eta_R,
                         I,
                         model_parameters_list,
                         M,
                         max_K, 
                         NL, 
                         mixing_resample, 
                         thresh_resample,
                         minimum_size,
                         do_density_est,
                         method); 
 
 //Implement SMC to sample from the trees' posterior
 my_particles.smc();

 //Given the result, compute the posteriror of the state variables
 //At the same time, if necessary, compute the posterior predictive densities
 my_particles.post_process();

 //Obtain the result of the posterior estimation
 List out = my_particles.output_result();
 
 //Cleaning
 my_particles.clear();

 return out;
}
