#ifndef RECURSION_H
#define RECURSION_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

class class_smc{
  
public:
  mat X; //data (n by d matrix)
  int G; //number of groups
  ivec groups_input; //groups labels
  mat grid_points; //Matrix of grid points for density estimation with d columns 
  ivec groups_pred_input; //groups labels for grid points
  double eta_R; //Now F(A) is shifted from 0 to 1 with prob 1- eta
  int I; //number of states
  List model_parameters_list; //tuning parameters specific to the model
  int M; // the number of particles
  int max_K; //The maximum resolution
  int NL; //The // of grid points for L
  double mixing_resample; //The parameter that controls the mixing of the resample steps
  double thresh_resample; //The parameter that determines the threshold in the resampling steps
  const int minimum_size; //Partitioning is terminated if # observations is less than minimum_size
  int do_density_est; //If=1, implement the density estimation. Otherwise, skip the step.
  int method; //1: APT, 2: MRS, 3: Other(custom)
  
  int n; //total sample size
  int d; //dimension
  
//Constructor
class_smc(const mat X, //data (n by d matrix)
           const int G, //number of groups
           const ivec groups_input, //groups labels
           const mat grid_points, //Matrix of grid points for density estimation with d columns 
           const ivec groups_pred_input, //groups labels for grid points
           const double eta_R, //Now F(A) is shifted from 0 to 1 with prob 1- eta
           const int I, //number of states
           const List model_parameters_list, //tuning parameters specific to the model
           const int M, // the number of particles
           const double max_K, //The maximum resolution
           const int NL, //The // of grid points for L
           const double mixing_resample, //The parameter that controls the mixing of the resample steps
           const double thresh_resample, //The parameter that determines the threshold in the resampling steps
           const int minimum_size, //Partitioning is terminated if # observations is less than minimum_size
           const int do_density_est, //If=1, implement the density estimation. Otherwise, skip the step
           const int method //1: APT, 2: MRS, 3: Other(custom)
);


private:
  ivec groups;
  ivec groups_pred;
  
  int n_location_options;
  
  int n_grid_points;
  
  cube partition_l_particle; //Left boundaries
  cube partition_r_particle; //Right boundaries
  cube log_post_state_particle; //Log of poseriors of the state variables
  imat obs_belong_ID_particle; //IDs to which each observation belongs
  icube n_obs_particle; //# observation (connsisting of vectors with G elements)
  imat parent_IDs_particle; //IDs of parent node
  
  imat D_particle;
  mat  L_particle;
  imat R_particle;
  imat k_particle;
  
  //Vecotors and matrices to record the progress of the particle filtering
  ivec n_generated_nodes;
  imat active_nodes;
  ivec n_active_nodes;
  
  //Matrix to record which particle makes which node.
  imat ancestor_indices_particle;

public:
  void init();
  void make_particles();
  
private:
  cube xi_cube; //cube that stores xi for each level
    
public:
  void make_xi_cube();
  
  void smc();
  void get_info_of_current_node(const int& index_p ,const int& node_ID_current);
  void construct_proposal();
  void compute_post_cubes();
  
  void make_count_matrices(const int& R);
  void make_unnormalized_post_cube(const icube& count_cube_l, const icube& count_cube_r, const vec& L_points, const vec& log_L_prior,
                                   const int& num_options_location, const int& R);
  void sample_from_proposal();
  void update_particles(const int& index_p ,const int& node_ID_current);
  void resample_particles();
  
  void update_particle_size();
  
  void post_process();
  
  void get_info_of_current_particle(const int& index_p);
  void compute_state_posterior();
  void compute_predictive_density(const int index_p);
  void compute_post_null_MRS(const int index_p);
  void compute_eff_MRS(const int index_best_tree);

  void compute_log_volumes();
  List output_result();
  
  void clear();
  
//Variables used in the smc update
private:
  double log_prob_refix;
  double log_prob_not_refix;
  
  vec L_points;
  vec log_L_prior;
  vec L_points_simple;
  vec log_L_prior_simple;
  
  vec log_post_state_prev;
  vec log_post_state_current;
  int k_current;
  mat xi_current;
  vec left_current;
  vec right_current;
  int R_current;
  
  uvec indices_X_current;
  mat X_current;
  ivec groups_current;
  ivec n_each_group_current;
  int n_current;
  
  icube count_cube_temp_0;
  icube count_cube_l_0;
  icube count_cube_r_0;

  icube count_cube_l_1;
  icube count_cube_r_1;
  
  cube unnormalized_post_cube_0;
  cube unnormalized_post_cube_1;
  
  int R_sample;
  int D_sample;
  int index_L_sample;
  double L_sample;
  
  vec log_incw_vec;
  
  //Other variables
  vec normalized_w;
  vec log_normalized_w;
  
  int memory_current;
  
  int node_ID_current_pre, node_ID_current_new;
  int ancestor_index_pre, ancestor_index_new; 
  
  //Variables to store the proposal distributions
  double log_incw_current;
  vec R_post;
  vec D_post_0, D_post_1;
  mat L_posterior_mat_0;
  
  
  //Variables used after sampling with the smc.
  int n_generated_nodes_i;
  ivec parent_IDs_i;
  mat partition_l_i, partition_r_i;
  ivec D_i;
  vec L_i;
  ivec k_i;
  imat n_obs_i;
  
  vec log_volume_i;
  
  imat children_IDs_i;
  
  vec log_L_densities;
  mat log_phi_post;
  mat log_Phi_post;
  cube xi_post;
  mat gamma_post;
  mat log_Q_post;
  mat q_post;
  
  //Lists to output the results.
  List tree_l_list;
  List tree_r_list;
  List gamma_post_list;
  List xi_list;
  List children_ID_list;
  vec pred_densities;

  vec log_L_joint_density_store;
  vec log_Phi_omega_store;
  mat log_phi_omega_store;
  
  vec post_null_MRS_store;
  vec eff_MAP;
  
  vec post_trees;
  int index_best_tree;
  
  List out;
};

#endif