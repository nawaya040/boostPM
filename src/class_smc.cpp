// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "helpers.h"
//#include "APT_functions.h"
//#include "MRS_functions.h"
#include "Boosting_functions.h"
#include "class_smc.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

#define MEMORY_LIMIT 10 //Initial number of nodes whose information is contained in each particle

#define EFF_SAMPLE 1000 //Sample size of Monte Carlo to approximate eff(A) in the estimation with MRS

#define SMALL_VALUE 1e-300
#define LARGE_VALUE 1e+300

class_smc::class_smc(const mat X,
           const int G,
           const ivec groups_input,
           const mat grid_points,
           const ivec groups_pred_input,
           const double eta_R,
           const int I,
           const List model_parameters_list,
           const int M,
           const double max_K,
           const int NL,
           const double mixing_resample,
           const double thresh_resample,
           const int minimum_size,
           const int do_density_est,
           const int method
):
  X(X),
  G(G),
  groups_input(groups_input),
  grid_points(grid_points),
  groups_pred_input(groups_pred_input),
  eta_R(eta_R),
  I(I),
  model_parameters_list(model_parameters_list),
  M(M),
  max_K(max_K),
  NL(NL),
  mixing_resample(mixing_resample),
  thresh_resample(thresh_resample),
  minimum_size(minimum_size),
  do_density_est(do_density_est),
  method(method)
{
  n = X.n_rows; //Total sample size
  d = X.n_cols; //Dimension
  
  n_location_options = NL-1;
  
  //Change the group labels so that they start from 0 instead of 1
  groups = groups_input - 1;
  groups_pred = groups_pred_input - 1;
  
  init();
}

void class_smc::init(){ //Initialization
  
  make_xi_cube();
  make_particles();
  
  //Precomputing spesicic to the model
  //if(method == 1){
  //  pre_compute_APT(G, I, NL, model_parameters_list);
  //}else if(method == 2){
  //  pre_compute_MRS(G, I, NL, model_parameters_list);
  //}else{
    pre_compute_Boosting(G, I, NL, model_parameters_list);
  //}
  
  
  //Make the grid ponits the prior for L
  L_points.resize(n_location_options);
  for(int i=0; i<n_location_options; i++){
    L_points(i) = (double) (i+1) / (double) (n_location_options+1);
  }
  
  L_points_simple = ones(1) * 0.5;
  log_L_prior_simple = zeros(1);
  
  //Make necessary vectors matrices, and ubes
  log_post_state_current.resize(I);
  
  count_cube_temp_0.resize(d, NL, G);
  count_cube_l_0.resize(d, n_location_options, G);
  count_cube_r_0.resize(d, n_location_options, G);
  
  count_cube_l_1.resize(d, 1, G);
  count_cube_r_1.resize(d, 1, G);
  
  unnormalized_post_cube_0.resize(d,n_location_options,I);
  unnormalized_post_cube_1.resize(d,1,I);
  
  n_grid_points = grid_points.n_rows;
}

void class_smc::make_xi_cube(){ //Pre-compute xi
  
  xi_cube = zeros(I,I,max_K);
  
  for(int k=0;k<max_K;k++){
    
    //if(method == 1){
    //  xi_cube.slice(k) = CreateXi_APT(I, model_parameters_list, k);
    //}else if(method == 2){
    //  xi_cube.slice(k) = CreateXi_MRS(I, model_parameters_list, k);
    //}else{
      xi_cube.slice(k) = CreateXi_Boosting(I, model_parameters_list, k);
    //}
  }
}

void class_smc::make_particles(){ //Initialize the particle matrices and cubes
  
  partition_l_particle.reshape(M, MEMORY_LIMIT, d);
  partition_r_particle.reshape(M, MEMORY_LIMIT, d);
  
  partition_l_particle.tube(0,0,M-1,0).fill(0);
  partition_r_particle.tube(0,0,M-1,0).fill(1);
  
  log_post_state_particle.reshape(M, MEMORY_LIMIT, I);
  mat xi_initial = xi_cube.slice(0);
  for(int i=0; i<M;i++){
    log_post_state_particle.tube(i,0) = log(xi_initial.row(0));
  }
  
  obs_belong_ID_particle.reshape(M, n);
  obs_belong_ID_particle.fill(0);
  
  n_obs_particle.reshape(M, MEMORY_LIMIT, G);
  ivec count_groups_initial = Count_groups(G, groups);
  for(int i=0; i<M;i++){
    n_obs_particle.tube(i,0) = count_groups_initial;
  }
  
  parent_IDs_particle.reshape(M, MEMORY_LIMIT);
  parent_IDs_particle.col(0).fill(-1); //This operation is just to find an error if it exists
  
  
  D_particle.reshape(M, MEMORY_LIMIT);
  D_particle.col(0).fill(-1); //This operation is just to find an error if it exists
  
  L_particle.reshape(M, MEMORY_LIMIT);
  L_particle.col(0).fill(-1.0); //This operation is just to find an error if it exists
  
  R_particle.reshape(M, MEMORY_LIMIT);
  if(eta_R == 0){
    R_particle.col(0).fill(1);
  }else{
    R_particle.col(0).fill(0);
  }
  
  k_particle.reshape(M, MEMORY_LIMIT);
  k_particle.col(0).fill(0);
  
  active_nodes.reshape(M, MEMORY_LIMIT);
  active_nodes.col(0).fill(1);
  
  n_active_nodes.resize(M);
  n_active_nodes.fill(1);
  
  n_generated_nodes.resize(M);
  n_generated_nodes.fill(1);
  
  ancestor_indices_particle.reshape(M, MEMORY_LIMIT);
  ancestor_indices_particle.col(0).fill(0);
}


void class_smc::smc(){ //Implement the SMC algorithm
  int terminate_or_not = 0;
  
  normalized_w = ones(M) / (double) M;
  log_normalized_w = log(normalized_w);
  
  memory_current = MEMORY_LIMIT;
  
  do{
    //Initialize the incremental weight vector
    log_incw_vec = zeros(M);
    
    //Find indices of particles that are to be updated.
    uvec indices_updated = find(n_active_nodes > 0);
    
    for(uword i=0; i<indices_updated.n_rows; i++){
      
      //Index of the particle updated here
      int index_p = (int) indices_updated(i);
      //The correspoinding node ID
      int node_ID_current = find(active_nodes.row(index_p) == 1).min();
      
      //Input the information of the current node
      get_info_of_current_node(index_p ,node_ID_current);
      
      //Check if the proposal distribution for the current node is alreadly computed or not.
      int new_or_not;
      if(i == 0){
        new_or_not = 1;
        
        node_ID_current_pre = node_ID_current;
        ancestor_index_pre = ancestor_indices_particle(index_p, node_ID_current);
      }else{
        node_ID_current_new = node_ID_current;
        ancestor_index_new = ancestor_indices_particle(index_p, node_ID_current);
        
        if((ancestor_index_pre == ancestor_index_new) && (node_ID_current_pre == node_ID_current_new)){
          new_or_not = 0;
        }else{
          new_or_not = 1;
          
          node_ID_current_pre = node_ID_current_new;
          ancestor_index_pre = ancestor_index_new;
        }
      }
      
      //Get the proposal distribution.
      if(new_or_not == 1){
        construct_proposal();
      }
      
      //Sample R, D, and L.
      sample_from_proposal();
      
      //Based on the sampled values, update the particles
      update_particles(index_p ,node_ID_current);
      
      //Input the incremental weight
      log_incw_vec(index_p) = log_incw_current;
    }
    
    //Compute the importance weights
    normalized_w = Normalize_log(log_normalized_w + log_incw_vec);
    log_normalized_w = log(normalized_w + SMALL_VALUE);
    
    //If ESS < Threshold, resample the particles
    double ESS = ComputeESS(normalized_w);
    
    if(ESS < thresh_resample * (double) M){
      resample_particles();
    }
    
    //If there is no active nodes left, finish the SMC
    if(all(n_active_nodes == 0)){
      terminate_or_not = 1;
    }
    
    //Finally, if the size of the particles is not enough, update their size.
    int max_n_generalted_nodes = arma::max(n_generated_nodes);
    
    if(max_n_generalted_nodes > memory_current-2){
      update_particle_size();
    }
    
  }while(terminate_or_not == 0);
  
}


void class_smc::get_info_of_current_node(const int& index_p ,const int& node_ID_current){
  
  //Posterior of states
  arma::vec log_post_state_prev = log_post_state_particle.tube(index_p, node_ID_current);
  
  //Current depth
  k_current = k_particle(index_p,node_ID_current);
  
  //Compute the transition matrix, with which we compute the posterior of states
  xi_current = xi_cube.slice(k_current);
  for(int j=0; j<I ;j++){
    log_post_state_current(j) = ColSum_log(log_post_state_prev + xi_current.col(j));
  }
  
  //The boundaries of the current node
  left_current = partition_l_particle.tube(index_p,node_ID_current);
  right_current = partition_r_particle.tube(index_p,node_ID_current);
  
  //R
  R_current = R_particle(index_p,node_ID_current);
  
  //Indices of the observations included in the current node
  indices_X_current = find(obs_belong_ID_particle.row(index_p) == node_ID_current);
  X_current = X.rows(indices_X_current);
  
  //Group labels
  groups_current = groups(indices_X_current);
  
  //Size of each group
  n_each_group_current = Count_groups(G, groups_current);
  
  //Size of the observations in the current node
  n_current = sum(n_each_group_current);
}


void class_smc::construct_proposal(){
  
  //Compute the prior of R == 0
  if(R_current == 0){
    vec log_L_prior_all0 = - abs(L_points - 0.5) * (eta_R * (double) n_current);
    vec log_L_prior_all = log_L_prior_all0 - ColSum_log(log_L_prior_all0);
    
    vec log_L_prior0 = log_L_prior_all;
    log_L_prior0(NL/2-1) = - LARGE_VALUE;
    log_prob_not_refix = ColSum_log(log_L_prior0);
    log_L_prior = log_L_prior0 - log_prob_not_refix;
    
    //Compute the probability of re-fixing the location variable
    log_prob_refix = log_L_prior_all(NL/2-1);
  }
  
  //Compute cubes that store the unnormalized posteriors
  compute_post_cubes();
  
  //1. Posteriors of R and D
  //At the same time, compute the incremental weight, which is in this case independent of the generated values
  vec log_D_post0_0(d);
  vec log_D_post0_1(d);
  
  for(int j=0;j<d;j++){
    log_D_post0_1(j) = Sum_log(unnormalized_post_cube_1.row(j));
  }
  
  D_post_1 = Normalize_log(log_D_post0_1);
  
  if(R_current == 1){
    
    log_incw_current = ColSum_log(log_D_post0_1);
    
  }else if(R_current == 0){
    
    vec log_R_post0(2);
    R_post = zeros(2);
    
    for(int j=0;j<d;j++){
      log_D_post0_0(j) = Sum_log(unnormalized_post_cube_0.row(j));
    }
    
    D_post_0 = Normalize_log(log_D_post0_0);
    
    log_R_post0[0] = log_prob_not_refix + ColSum_log(log_D_post0_0);
    log_R_post0[1] = log_prob_refix + ColSum_log(log_D_post0_1);
    
    R_post = Normalize_log(log_R_post0);
    
    log_incw_current = ColSum_log(log_R_post0);
  }
  
  //2. Posterior of L
  if(R_current == 0){
    L_posterior_mat_0 = zeros(d, n_location_options);
    for(int j=0;j<d;j++){
      
      vec log_L_post_j = colsum_for_cube(unnormalized_post_cube_0.row(j), n_location_options);
      vec L_post_j = Normalize_log(log_L_post_j);
      
      L_posterior_mat_0.row(j) = L_post_j.t();
    }
  }
  
}

void class_smc::sample_from_proposal(){
  
  //1. R
  if(R_current == 1){
    R_sample = 1;
  }else if(R_current == 0){
    R_sample = OneSample(R_post);
  }
  
  //2. D
  if(R_sample == 1){
    D_sample = OneSample(D_post_1);
  }else if(R_sample == 0){
    D_sample = OneSample(D_post_0);
  }
  
  
  //3. L
  if(R_sample == 1){
    index_L_sample = 0;
    L_sample = 0.5;
  }else if(R_sample == 0){
    vec L_post = L_posterior_mat_0.row(D_sample).t();
    
    index_L_sample = OneSample(L_post);
    L_sample = L_points(index_L_sample);
  }
}

void class_smc::compute_post_cubes(){
  
  if(R_current == 0){
    make_count_matrices(0);
    
    make_unnormalized_post_cube(count_cube_l_0,count_cube_r_0,L_points,log_L_prior,n_location_options,0);
  }
  
  make_count_matrices(1);
  make_unnormalized_post_cube(count_cube_l_1,count_cube_r_1,L_points_simple,log_L_prior_simple,1,1);
  
}

void class_smc::make_count_matrices(const int& R){ //Count the number of observations included in the (possible) children nodes
  
  double left_j, right_j, middle_j;
  
  switch(R){
  
  case 0:
    
    count_cube_temp_0.fill(0);
    
    //Count the observations left/right to the partition points.
    for(int j=0;j<d;j++){
      left_j = left_current(j);
      right_j = right_current(j);
      
      for(int l=0;l<n_current;l++){
        int group_l = groups_current(l);
        int position_index = get_position_index(X_current(l,j), left_j, right_j, NL);
        
        count_cube_temp_0.subcube(j,position_index,group_l,j,NL-1,group_l) += 1;
      }
    }
    
    count_cube_l_0 = count_cube_temp_0.cols(0, n_location_options-1);
    
    for(int g=0;g<G;g++){
      count_cube_r_0.slice(g) = n_each_group_current(g) - count_cube_l_0.slice(g);
    }
    
  case 1:
    
    count_cube_l_1.fill(0);
    count_cube_r_1.fill(0);
    
    //Count the observations left/right to the partition points.
    for(int j=0;j<d;j++){
      left_j = left_current(j);
      right_j = right_current(j);
      middle_j = (left_j+right_j)/2;
      
      for(int l=0;l<n_current;l++){
        int group_l = groups_current(l);
        
        if(X_current(l,j) < middle_j){
          count_cube_l_1(j,0,group_l) += 1;
        }else{
          count_cube_r_1(j,0,group_l) += 1;
        }
        
      }
    }
  }
  
}


void class_smc::make_unnormalized_post_cube(const icube& count_cube_l, const icube& count_cube_r, const vec& L_points, const vec& log_L_prior,
                                            const int& n_location_options, const int& R){
  
  vec log_density_ratio(n_location_options);
  vec log_ML(n_location_options);
  
  ivec count_l_j(n_location_options);
  ivec count_r_j(n_location_options);
  
  for(int j=0;j<d;j++){
    
    log_density_ratio = colsum_for_icube(count_cube_l.row(j), n_location_options) % log(pow(L_points, -1.0)) +
      colsum_for_icube(count_cube_r.row(j), n_location_options) % log(pow(1 - L_points, -1.0));
    
    //V(A) = 1,..., I-1, I
    for(int v=1;v<I+1;v++){
      
      for(int l=0;l<n_location_options;l++){
        
        //if(method == 1){
        //  log_ML(l) = log_ML_compute_APT(G, I, count_cube_l.tube(j,l), count_cube_r.tube(j,l), L_points(l), NL, v, k_current);
        //}else if(method == 2){
        //  log_ML(l) = log_ML_compute_MRS(G, I, count_cube_l.tube(j,l), count_cube_r.tube(j,l), L_points(l), NL, v, k_current);
        //}else{
          log_ML(l) = log_ML_compute_Boosting(G, I, count_cube_l.tube(j,l), count_cube_r.tube(j,l), L_points(l), NL, v, k_current);
        //}
      }
      
      if(R == 0){
        unnormalized_post_cube_0.slice(v-1).row(j) =  log_post_state_current(v-1) - log((double) d) + log_L_prior.t() + log_ML.t() + log_density_ratio.t();
      }else if(R == 1){
        unnormalized_post_cube_1.slice(v-1).row(j) =  log_post_state_current(v-1) - log((double) d) + log_L_prior.t() + log_ML.t() + log_density_ratio.t();
      }
      
    }
  }
}


void class_smc::update_particles(const int& index_p ,const int& node_ID_current){
  
  int left_child_node_ID = n_generated_nodes(index_p);
  int right_child_node_ID = left_child_node_ID + 1;
  
  //The partition point in the chosen dimension
  double partition_sample = left_current(D_sample) + L_sample * (right_current(D_sample) - left_current(D_sample));
  
  //New left subnode
  vec left_new_l = left_current;
  vec right_new_l = right_current;
  
  right_new_l(D_sample) = partition_sample;
  
  partition_l_particle.tube(index_p, left_child_node_ID) = left_new_l;
  partition_r_particle.tube(index_p, left_child_node_ID) = right_new_l;
  
  //New right subnode
  vec left_new_r = left_current;
  vec right_new_r = right_current;
  
  left_new_r(D_sample) = partition_sample;
  
  partition_l_particle.tube(index_p, right_child_node_ID) = left_new_r;
  partition_r_particle.tube(index_p, right_child_node_ID) = right_new_r;
  
  //log_post_state
  vec log_post_state_new0(I);
  
  if(R_sample == 1){
    log_post_state_new0 = unnormalized_post_cube_1.tube(D_sample, index_L_sample);
  }else if(R_sample == 0){
    log_post_state_new0 = unnormalized_post_cube_0.tube(D_sample, 0);
  }
  
  vec log_post_state_new = log(Normalize_log(log_post_state_new0));
  
  log_post_state_particle.tube(index_p, left_child_node_ID) = log_post_state_new;
  log_post_state_particle.tube(index_p,right_child_node_ID) = log_post_state_new;
  
  //obs_belong_ID_particle and n_obs_particle
  ivec n_left_child = zeros<ivec>(G);
  ivec n_right_child =zeros<ivec>(G);
  
  for(int i=0;i<n_current;i++){
    int index_temp = indices_X_current(i);
    
    if(X_current(i,D_sample) <= partition_sample){
      
      //This observation goes into the left node.
      n_left_child(groups_current(i)) += 1;
      obs_belong_ID_particle(index_p, index_temp) = left_child_node_ID;
      
    }else{
      
      //This observation goes into the left node.
      n_right_child(groups_current(i)) += 1;
      obs_belong_ID_particle(index_p, index_temp) = right_child_node_ID;
      
    }
  }
  
  n_obs_particle.tube(index_p,left_child_node_ID) = n_left_child;
  n_obs_particle.tube(index_p,right_child_node_ID) = n_right_child;
  
  //parent
  parent_IDs_particle(index_p, left_child_node_ID) = node_ID_current;
  parent_IDs_particle(index_p, right_child_node_ID) = node_ID_current;
  
  //D
  D_particle(index_p, left_child_node_ID) = D_sample;
  D_particle(index_p,right_child_node_ID) = D_sample;
  
  //L
  L_particle(index_p, left_child_node_ID) = L_sample;
  L_particle(index_p,right_child_node_ID) = L_sample;
  
  //F
  R_particle(index_p, left_child_node_ID) = R_sample;
  R_particle(index_p,right_child_node_ID) = R_sample;
  
  //k
  int k_new = k_current+1;
  k_particle(index_p, left_child_node_ID) = k_new;
  k_particle(index_p,right_child_node_ID) = k_new;
  
  //n_generated_nodes
  n_generated_nodes(index_p) += 2;
  
  //active_nodes and n_active_nodes
  //left node
  int div_or_not_l = (sum(n_left_child) >= minimum_size) && (k_new < max_K);
  
  if(div_or_not_l == 1){
    active_nodes(index_p, left_child_node_ID) = 1;
    n_active_nodes(index_p) += 1;
    
    ancestor_indices_particle(index_p, left_child_node_ID) = index_p;
  }
  //right node
  int div_or_not_r = (sum(n_right_child) >= minimum_size) && (k_new < max_K);
  
  if(div_or_not_r == 1){
    active_nodes(index_p,right_child_node_ID) = 1;
    n_active_nodes(index_p) += 1;
    
    ancestor_indices_particle(index_p, right_child_node_ID) = index_p;
  }
  
  
  //Finally, since the current node is no longer active, we need to input this information.
  active_nodes(index_p, node_ID_current) = 0;
  n_active_nodes(index_p) -= 1;
}


void class_smc::resample_particles(){
  
  //Compute the weights to resample the particles
  vec w_for_resample0 = pow(normalized_w,mixing_resample);
  vec w_for_resample = w_for_resample0 / sum(w_for_resample0);
  
  uvec resample_parent = Resample_index(M,w_for_resample);
  
  //Resample the particles
  Resample_cube(M,partition_l_particle,resample_parent);
  Resample_cube(M,partition_r_particle,resample_parent);
  
  Resample_cube(M,log_post_state_particle,resample_parent);
  
  obs_belong_ID_particle = obs_belong_ID_particle.rows(resample_parent);
  
  Resample_icube(M,n_obs_particle,resample_parent);
  
  parent_IDs_particle = parent_IDs_particle.rows(resample_parent);
  
  D_particle = D_particle.rows(resample_parent);
  L_particle = L_particle.rows(resample_parent);
  R_particle = R_particle.rows(resample_parent);
  k_particle = k_particle.rows(resample_parent);
  
  n_generated_nodes = n_generated_nodes.rows(resample_parent);
  active_nodes = active_nodes.rows(resample_parent);
  n_active_nodes = n_active_nodes.rows(resample_parent);
  
  ancestor_indices_particle = ancestor_indices_particle.rows(resample_parent);
  
  //Update log_vw
  vec w_new0 = pow(normalized_w,1.0-mixing_resample);
  
  log_normalized_w = log(w_new0 / sum(w_new0) + SMALL_VALUE);
}


void class_smc::update_particle_size(){
  
  int memory_prev = memory_current;
  memory_current = memory_current * 2;
  
  cube partition_l_particle_old = partition_l_particle;
  cube partition_r_particle_old = partition_r_particle;
  partition_l_particle = zeros(M,memory_current,d);
  partition_r_particle = zeros(M,memory_current,d);
  partition_l_particle.cols(0,memory_prev-1) = partition_l_particle_old;
  partition_r_particle.cols(0,memory_prev-1) = partition_r_particle_old;
  
  cube log_post_state_particle_old = log_post_state_particle;
  log_post_state_particle = zeros(M,memory_current,I);
  log_post_state_particle.cols(0,memory_prev-1) = log_post_state_particle_old;
  
  icube n_obs_particle_old = n_obs_particle;
  n_obs_particle = zeros<icube>(M,memory_current,G);
  n_obs_particle.cols(0,memory_prev-1) = n_obs_particle_old;
  
  parent_IDs_particle.reshape(M,memory_current);
  
  D_particle.reshape(M,memory_current);
  L_particle.reshape(M,memory_current);
  R_particle.reshape(M,memory_current);
  k_particle.reshape(M,memory_current);
  
  active_nodes.reshape(M,memory_current);
  
  ancestor_indices_particle.reshape(M,memory_current);
}

void class_smc::post_process(){
  
  log_L_joint_density_store = zeros(M);
  
  log_Phi_omega_store = zeros(M);
  log_phi_omega_store = zeros(M, I);
  pred_densities = zeros(n_grid_points);
  
  post_null_MRS_store = zeros(M);
  
  for(int index_p=0;index_p<M;index_p++){
    //Get the necessary information from the generated particle
    get_info_of_current_particle(index_p);
    
    //Compute the posterior distribution of the state variables
    compute_state_posterior();
    
    //Input the information necessary to compute the marginal likelihood of the model
    log_Phi_omega_store(index_p) = log_Phi_post(0,0);
    
    for(int i=0;i<I;i++){
      log_phi_omega_store(index_p,i) = log_Phi_post(i,0);
    }
    
    log_L_joint_density_store(index_p) = sum(log_L_densities);
    
    if(do_density_est == 1){
      //If necessary, compute the poserior predictive density.
      compute_predictive_density(index_p);
    }
    
    if(method == 2){
      compute_post_null_MRS(index_p);
    }
  }
  
  //Find the MAP tree
  vec log_post_trees0 = log_L_joint_density_store + log_Phi_omega_store; //We can ignore the posterior of D because it is uniform
  post_trees = Normalize_log(log_post_trees0);
  index_best_tree = post_trees.index_max();
  
  if(method == 2){
    compute_eff_MRS(index_best_tree);
  }
}

void class_smc::get_info_of_current_particle(const int& index_p){
  //# total nodes
  n_generated_nodes_i = n_generated_nodes(index_p);
  
  //Get the necessary information from the particle system
  ivec parent_IDs_i0 = parent_IDs_particle.row(index_p).t();
  parent_IDs_i = parent_IDs_i0.subvec(0, n_generated_nodes_i-1);
  
  mat partition_l_i0 = partition_l_particle.row(index_p);
  mat partition_r_i0 = partition_r_particle.row(index_p);
  
  if(d > 1){
    partition_l_i0 = partition_l_i0.t();
    partition_r_i0 = partition_r_i0.t();
  }
  
  partition_l_i = partition_l_i0.cols(0, n_generated_nodes_i-1);
  partition_r_i = partition_r_i0.cols(0, n_generated_nodes_i-1);
  
  //Get the information of trees
  tree_l_list.push_back(partition_l_i);
  tree_r_list.push_back(partition_r_i);
  
  D_i = D_particle.row(index_p).t();
  L_i = L_particle.row(index_p).t();
  k_i = k_particle.row(index_p).t();
  
  n_obs_i.reshape(G,n_generated_nodes_i);
  for(int g=0;g<G;g++){
    ivec n_obs_i_g(n_generated_nodes_i);
    
    for(int j=0;j<n_generated_nodes_i;j++){
      n_obs_i_g(j) = n_obs_particle(index_p,j,g);
    }
    
    n_obs_i.row(g) = n_obs_i_g.t();
  }
  
  //Find IDs of each nodes' children
  children_IDs_i.reshape(2, n_generated_nodes_i);
  children_IDs_i.fill(-1); //-1 means the node is a terminal one
  
  children_IDs_i(0,0) = 1;
  children_IDs_i(1,0) = 2;
  
  for(int j=1; j<n_generated_nodes_i; j++){
    uvec children_IDs_temp = find(parent_IDs_i == j);
    
    //If a node has children, its number is always 2.
    if(children_IDs_temp.n_rows > 0){
      children_IDs_i.col(j) = conv_to< ivec >::from(children_IDs_temp);
    }
  }
  
  //Store the information
  children_ID_list.push_back(children_IDs_i);
  
  //Compute the volumes of the nodes
  compute_log_volumes();
}


void class_smc::compute_log_volumes(){
  log_volume_i = zeros(n_generated_nodes_i);
  
  for(int j=0; j<n_generated_nodes_i; j++){
    log_volume_i(j) = sum(log(partition_r_i.col(j) - partition_l_i.col(j)));
  }
}



void class_smc::compute_state_posterior(){
  
  //1. Compute phi and Phi (for the defitinon, see the manuscript)
  //  At the same time, compute the density function of the sampled L, which is necessary to find the MAP tree
  
  log_phi_post = zeros(I,n_generated_nodes_i);
  log_Phi_post = zeros(I,n_generated_nodes_i);
  xi_post = zeros(I,I,n_generated_nodes_i);
  
  log_phi_post.fill(-1);
  log_Phi_post.fill(-1);
  
  log_L_densities = zeros(n_generated_nodes_i);
  
  for(int j=n_generated_nodes_i-1;j>=0;j--){
    
    int k_j = k_i(j);
    mat xi_j =  xi_cube.slice(k_current);
    
    if(children_IDs_i(0,j) == -1){
      
      //Terminal node
      log_Phi_post.col(j) = sum(n_obs_i.col(j)) * (-log_volume_i(j)) * ones(I,1);
      //xi_post is just the same as the prior.
      xi_post.slice(j) = xi_j;
      
    }else{
      
      //Non-terminal node
      int child_ID_l = children_IDs_i(0,j);
      int child_ID_r = children_IDs_i(1,j);
      double L_child = L_i(child_ID_l);
      int L_child_index = (L_child * NL)-1;
      
      //Compute phi
      for(int v=1;v<(I+1);v++)
      {
        
        //if(method == 1){
          
          //if(v < I){
          ////The current node is a non-leaf node
        //  log_phi_post(v-1,j) = log_ML_compute_APT(G, I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_child, NL, v, k_j) +
        //    log_Phi_post(v-1,child_ID_l) + log_Phi_post(v-1,child_ID_r);
          //}else{
          ////The current node is a leaf node
          //  log_phi_post(I-1,j) =  sum(n_obs_i.col(j)) * (-log_volume_i(j));
          //}
          
        //}else if(method == 2){
        //  log_phi_post(v-1,j) = log_ML_compute_MRS(G, I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_child, NL, v, k_j) +
        //    log_Phi_post(v-1,child_ID_l) + log_Phi_post(v-1,child_ID_r);
        //}else{
          log_phi_post(v-1,j) = log_ML_compute_Boosting(G, I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_child, NL, v, k_j) +
            log_Phi_post(v-1,child_ID_l) + log_Phi_post(v-1,child_ID_r);
        //}
        
      }
      
      //Compute Phi
      for(int v=1;v<I+1;v++){
        log_Phi_post(v-1,j) = ColSum_log(log(xi_j.row(v-1) + SMALL_VALUE).t() + log_phi_post.col(j));
      }
      
      //compute xi_post
      for(int v=1;v<I+1;v++){
        xi_post.slice(j).row(v-1) = Normalize_log(log(xi_j.row(v-1) + SMALL_VALUE).t() + log_phi_post.col(j)).t();
      }
      
      //Compute the density of the sampled L
      int n_temp = sum(n_obs_i.col(j));
      vec log_L_prior_all0 = - abs(L_points - 0.5) * (eta_R * (double) n_temp);
      vec log_L_prior_all = log_L_prior_all0 - ColSum_log(log_L_prior_all0);
      
      log_L_densities(j) = log_L_prior_all(L_child_index);
    }
  }
  
  //2. Based on the reuslts, compute the posteriors of states
  gamma_post = zeros(I, n_generated_nodes_i);
  gamma_post.col(0) = xi_post.slice(0).row(0).t();
  
  for(int j=1;j<n_generated_nodes_i;j++){
    int parent_ID_temp = parent_IDs_i(j);
    rowvec gamma_post_temp = gamma_post.col(parent_ID_temp).t() * xi_post.slice(j);
    
    gamma_post.col(j) = gamma_post_temp.t();
  }
  
  //Store the result
  xi_list.push_back(xi_post);
  gamma_post_list.push_back(gamma_post);
  
}



void class_smc::compute_predictive_density(const int index_p){
  
  log_Q_post = zeros(I, n_generated_nodes_i); //Each cell stores e_A(i) in the paper.
  q_post = zeros(G, n_generated_nodes_i);
  
  for(int g=0;g<G;g++){
    //Compute the probability for each node
    log_Q_post.col(0) = log(gamma_post.col(0) + SMALL_VALUE);
    
    for(int j=0;j<n_generated_nodes_i;j++){
      if(children_IDs_i(0,j) != -1){
        int child_ID_l = children_IDs_i(0,j);
        int child_ID_r = children_IDs_i(1,j);
        
        double L_temp = L_i(child_ID_l);
        
        int k_j = k_i(j);
        
        arma::vec mean_temp(I);
        
        for(int v=1;v<I+1;v++){
          //if(method == 1){
          //  mean_temp(v-1) = PostMean_APT(I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_temp, v, g, k_j);
          //}else if(method == 2){
          //  mean_temp(v-1) = PostMean_MRS(I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_temp, v, g, k_j);
          //}else{
            mean_temp(v-1) = PostMean_Boosting(I, n_obs_i.col(child_ID_l), n_obs_i.col(child_ID_r), L_temp, v, g, k_j);
          //}
        }
        
        //Left child
        for(int v=1;v<I+1;v++){
          vec temp_vec = log(xi_post.slice(child_ID_l).col(v-1) + SMALL_VALUE) + log(mean_temp) + log_Q_post.col(j);
          log_Q_post(v-1,child_ID_l) = ColSum_log(temp_vec);
        }
        
        //Right child
        for(int v=1;v<I+1;v++){
          vec temp_vec = log(xi_post.slice(child_ID_r).col(v-1) + SMALL_VALUE) + log(1-mean_temp) + log_Q_post.col(j);
          log_Q_post(v-1,child_ID_r) = ColSum_log(temp_vec);
        }
      }
    }
    
    //Comute the probability density
    for(int j=0;j<n_generated_nodes_i;j++){
      q_post(g,j) = exp(ColSum_log(log_Q_post.col(j)) - log_volume_i(j));
    }
    
  }
  
  //Compute the predictive density for each input point
  //Find a terminal node ID to which each grid point belongs
  ivec terminal_IDs = zeros<ivec>(n_grid_points);
  
  for(int j=0;j<n_generated_nodes_i;j++){
    if(children_IDs_i(0,j) != -1){
      //Get the indices of grid points that belong to the current node
      arma::uvec indices_current = find(terminal_IDs == j);
      
      //Get grid points that belong to the current node
      arma::mat grid_points_current(indices_current.n_rows,d);
      
      for(uword k=0; k<indices_current.n_rows; k++){
        grid_points_current.row(k) = grid_points.row(indices_current(k));
      }
      
      uword child_ID_l = children_IDs_i(0,j);
      uword child_ID_r = children_IDs_i(1,j);
      
      //Check to which child node each current grid point belongs
      double L_current = L_i(child_ID_l);
      int D_current =  D_i(child_ID_l);
      double partition_l_current = partition_l_i(D_current,j);
      double partition_r_current = partition_r_i(D_current,j);
      
      double partition_current = partition_l_current + L_current * (partition_r_current - partition_l_current);
      
      arma::uvec indices_l_child = indices_current(find(grid_points_current.col(D_current) <= partition_current));
      arma::uvec indices_r_child = indices_current(find(partition_current < grid_points_current.col(D_current)));
      
      //Left child
      for(uword k=0;k<indices_l_child.n_rows;k++){
        terminal_IDs(indices_l_child(k)) = child_ID_l;
      }
      
      //Right child
      for(uword k=0;k<indices_r_child.n_rows;k++){
        terminal_IDs(indices_r_child(k)) = child_ID_r;
      }
    }
  }
  
  
  vec pred_densities_current(n_grid_points);
  for(int l=0;l<n_grid_points;l++){
    pred_densities_current(l) =  q_post(groups_pred(l), terminal_IDs(l));
  }
  
  //Input the result
  pred_densities = pred_densities + exp(log_normalized_w(index_p)) * pred_densities_current;
  
}

void class_smc::compute_post_null_MRS(const int index_p){
  
  vec psi_tilde = zeros(n_generated_nodes_i);
  
  for(int j=n_generated_nodes_i-1;j>-1;j--){
    if(children_IDs_i(0,j) == -1){
      //Leaf node
      psi_tilde(j) = xi_post(1,1,j) + xi_post(1,2,j);
    }else{
      //Non-lead node
      int child_ID_l = children_IDs_i(0,j);
      int child_ID_r = children_IDs_i(1,j);
      
      //Notice that the operation is different for the root node because the initial state is 0
      if(j==0){
        psi_tilde(j) = xi_post(0,1,j) * psi_tilde(child_ID_l) * psi_tilde(child_ID_r) + xi_post(0,2,j);
      }else{
        psi_tilde(j) = xi_post(1,1,j) * psi_tilde(child_ID_l) * psi_tilde(child_ID_r) + xi_post(1,2,j);
      }
    }
  }
  
  post_null_MRS_store(index_p) = psi_tilde(0);
}

void class_smc::compute_eff_MRS(const int index_best_tree){
  
  //1. Obtain the information necessary to compute the eff
  //# total nodes
  int n_generated_nodes_MAP = n_generated_nodes(index_best_tree);
  mat children_IDs_MAP = children_ID_list(index_best_tree);
  mat gamma_post_MAP = gamma_post_list(index_best_tree);
  imat n_obs_MAP;
  
  n_obs_MAP.reshape(G,n_generated_nodes_MAP);
  for(int g=0;g<G;g++){
    ivec n_obs_MAP_g(n_generated_nodes_MAP);
    
    for(int j=0;j<n_generated_nodes_MAP;j++){
      n_obs_MAP_g(j) = n_obs_particle(index_best_tree,j,g);
    }
    
    n_obs_MAP.row(g) = n_obs_MAP_g.t();
  }
  
  vec L_MAP = L_particle.row(index_best_tree).t();
  
  double prec_theta = model_parameters_list["precision_theta"];
  
  //2. Compute the eff for each node
  eff_MAP = zeros(n_generated_nodes_MAP);
  eff_MAP.fill(-1.0);
  
  for(int j=0;j<n_generated_nodes_MAP;j++){
    if(children_IDs_MAP(0,j) != -1){
      
      //Input the information of the children nodes
      int child_ID_l = children_IDs_MAP(0,j);
      int child_ID_r = children_IDs_MAP(1,j);
      
      double L_temp = L_MAP(child_ID_l);
      
      ivec n_l_vec = n_obs_MAP.col(child_ID_l);
      ivec n_r_vec = n_obs_MAP.col(child_ID_r);
      
      mat theta_post_sample(EFF_SAMPLE,G);
      
      for(int g=0;g<G;g++){
        
        double alpha_l = prec_theta * L_temp + n_l_vec(g);
        double alpha_r = prec_theta * (1-L_temp) + n_r_vec(g);
        
        for(int i=0;i<EFF_SAMPLE;i++){
          theta_post_sample(i,g) = R::rbeta(alpha_l, alpha_r);
        }
      }
      
      eff_MAP(j) = gamma_post_MAP(0,j) * abs(
        mean(
          log(
            theta_post_sample.col(0) / (1-theta_post_sample.col(0) + SMALL_VALUE)
        / ( theta_post_sample.col(1) / (1-theta_post_sample.col(1) + SMALL_VALUE) ) )
        )
      );
      
    }
    
  }
  
}

List class_smc::output_result(){
  //Compute the posterior of V(Omega)
  vec log_phi_omega_ave(I);
  for(int i=0;i<I;i++){
    vec log_phi_omega_i = log_phi_omega_store.col(i);
    log_phi_omega_ave(i) = ColSum_log(log_phi_omega_i + log_normalized_w);
  }
  
  vec V_omega_post = Normalize_log(log_phi_omega_ave);
  
  mat tree_l = tree_l_list(index_best_tree);
  mat tree_r = tree_r_list(index_best_tree);
  
  mat children_IDs = children_ID_list(index_best_tree);
  
  cube xi_post = xi_list(index_best_tree);
  mat gamma_post = gamma_post_list(index_best_tree);
  
  //Report the result in the form of list.
  if(do_density_est == 1){
    out = Rcpp::List::create(Rcpp::Named("tree_left") = tree_l,
                             Rcpp::Named("tree_right") = tree_r,
                             Rcpp::Named("children_IDs") = children_IDs,
                             Rcpp::Named("posterior_states") = gamma_post,
                             Rcpp::Named("posterior_xi") = xi_post,
                             Rcpp::Named("pred_density") = pred_densities,
                             Rcpp::Named("V_omega_post") = V_omega_post);
  }else if(method == 2){
    double post_null = sum(exp(log_normalized_w) % post_null_MRS_store);
    
    out = Rcpp::List::create(Rcpp::Named("tree_left") = tree_l,
                             Rcpp::Named("tree_right") = tree_r,
                             Rcpp::Named("children_IDs") = children_IDs,
                             Rcpp::Named("posterior_states") = gamma_post,
                             Rcpp::Named("posterior_xi") = xi_post,
                             Rcpp::Named("V_omega_post") = V_omega_post,
                             Rcpp::Named("post_null") = post_null,
                             Rcpp::Named("eff_MAP") = eff_MAP);
  }else{
    out = Rcpp::List::create(Rcpp::Named("tree_left") = tree_l,
                             Rcpp::Named("tree_right") = tree_r,
                             Rcpp::Named("children_IDs") = children_IDs,
                             Rcpp::Named("posterior_states") = gamma_post,
                             Rcpp::Named("posterior_xi") = xi_post,
                             Rcpp::Named("V_omega_post") = V_omega_post);
  }
  
  return out;
}

void class_smc::clear(){
  
  partition_l_particle.clear();
  partition_r_particle.clear();
  log_post_state_particle.clear();
  obs_belong_ID_particle.clear();
  n_obs_particle.clear();
  parent_IDs_particle.clear();
  
  D_particle.clear();
  L_particle.clear();
  R_particle.clear();
  k_particle.clear();
  
  n_generated_nodes.clear();
  active_nodes.clear();
  n_active_nodes.clear();
  
  ancestor_indices_particle.clear();
}
