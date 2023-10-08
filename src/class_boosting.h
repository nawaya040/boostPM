#ifndef BOOST_H
#define BOOST_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

struct Node
{
  unsigned int node_id; //1(=root),2,3,...
  
  int depth;
  
  vec left_points;
  vec right_points;
  
  int dim_selected; 
  
  double location; // the value of "L"
  double partition_point; //The partition point in the selected dimension
  
  double precision;
  
  unsigned int counts;
  
  double theta_old;
  double theta;

  vector<int> indices; //indices of observations included in this node
  
  Node* parent = nullptr;
  Node* left = nullptr;
  Node* right = nullptr;
};

class class_boosting{
  
public:
  
  //Constructor
  class_boosting(
                      mat X,
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
  );
  
  //Input information
  mat X; //n x d 
  double precision;
  double alpha;
  double beta;
  double gamma;
  double rho; //what is rho?
  int max_resol;
  int num_each_dim;
  int num_second;

  double learn_rate;
  int min_obs;
  int nbins;
  
  double eta_subsample;
  double thresh_stop;
  int ntrees_wait;
  
  int max_n_var;

  int parameter_for_test;

  //variables
  int n;
  int d;
  //int n_eval;
  
  int num_trees;
  ivec active_vars;
  ivec vars_chosen;
  

  ///////////////////////////////////////////////////////////////////////////
  // variables for boosting
  mat residuals_current; //Note: this matrix is d x n
  
  int num_grid_points_L;
  vec L_candidates;
  
  int current_dim_first;
  mat log_like_matrix;
  
  Node** root_nodes;
  
  vector<int> tree_size_store;
  vector<int> max_depth_store;
  
  mat residuals_last_boosting;

  vec importances;
  
  bool is_first_stage;
  
  int size_subsample;
  
  uvec indices_used;
  uvec indices_not_used;
  
  vector<double> improvement_curve;
  
  //variables to store the information of generated trees
  vector<int> d_store;
  vector<double> l_store;
  vector<double> theta_store;
  List tree_list;
  
  //use only a part of variables
  ivec is_selected_vec;
  
  //Variables to store the information of the old tree and measure
  int dim_selected_old;
  double location_old;
  double partition_point_old;
  double theta_old;

  //Initialization
  void init();

  //tree functions
  Node* get_root_node();
  Node* get_new_node(Node* parent, bool this_is_left, int dim_selected, double location);

  void add_children(Node* node, int dim_selected, double location);

  void count_total_nodes(Node* node, int& count);
  
  Node* find_terminal_node(Node* root, vec& x);
  
  double evaluate_density(Node* root, vec& x);
  
  //Boosting functions
  void boosting();
  void construct_tree(Node* node);
  bool split_node(Node* node);
  
  //utilities for boosting
  double get_precision(Node* node);
  ivec make_left_count_vector(Node* node, int dim);
  double get_split_prob(Node* node);
  void check_max_depth( Node* node, int& depth_max);
  void print_progress_boosting(int step);

  vec residualize(Node* root, vec& x);
  
  double local_move(double x, double left_point, double right_point, 
                    double theta, double area_ratio, bool left);
  double evaluate_log_prior(Node* node);

  //print the progress of mcmc sampling
  void print_progress(int index_MCMC);
  
  //output
  List output();
  
  //destructor
  ~class_boosting();
  void clear_node(Node* root);
  
  //miscellaneous functions
  double compute_precision(int depth);
  double compute_splitting_prob(int depth);

};
  
#endif