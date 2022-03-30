// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include<bits/stdc++.h>
#include "class_boosting.h"
#include "count_tree.h"
#include "helpers.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

#define INDEX_ZERO 0
#define NUM_MOVES 2
#define LARGE_NUMBER 1e+100

#define MIN_WIDTH 1e-10

class_boosting::class_boosting(
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
                     int J,
                     double eta_subsample,
                     int max_n_var
):
  X(X),
  precision(precision),
  alpha(alpha),
  beta(beta),
  gamma(gamma),
  max_resol(max_resol),
  num_each_dim(num_each_dim),
  num_second(num_second),
  learn_rate(learn_rate),
  min_obs(min_obs),
  J(J),
  eta_subsample(eta_subsample),
  max_n_var(max_n_var)
{

  init();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Initialization
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void class_boosting::init(){ //Initialization
  
  //Input the basic information
  n = X.n_rows;
  d = X.n_cols;

  ///////////////////////////////////////////////////////////////////////////
  //initialization for boosting
  ///////////////////////////////////////////////////////////////////////////
  
  //Input the data as current residuals
  residuals_current = X.t();
  X.clear();
  
  //Make (another type of) tree used for counting
  ctree.init(J);
  
  num_trees = num_each_dim * d + num_second;
  
  
  //input possible values of L
  double gap = pow(2.0, (double) -J);
  num_grid_points_L = pow(2,J)-1;
  L_candidates = linspace(gap, 1- gap, num_grid_points_L);
  
  //make a matrix used to compute the posterior probabilities
  log_like_matrix = zeros(d, num_grid_points_L);

  //vectors to store the information of trees
  tree_size_store = zeros(num_trees);
  max_depth_store = zeros(num_trees);
  
  //vector to store the variable importance 
  importances = zeros(d);
  
  active_vars = zeros<ivec>(d);
  vars_chosen = zeros<ivec>(d);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//tree functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

Node* class_boosting::get_root_node(){
  Node* new_node = new Node;
  
  new_node->depth = 0;
  
  new_node->node_id = 1;
  
  new_node->left_points = zeros<vec>(d);
  new_node->right_points = ones<vec>(d);
  
  new_node->dim_selected = -1; 
  new_node->location = 0;
  new_node->partition_point = 0; 
  
  new_node->parent = nullptr;
  new_node->left = nullptr;
  new_node->right = nullptr;
  new_node->counts = 0;
  new_node->precision = compute_precision(new_node->depth);
  
  
  if(eta_subsample < 1.0){
    //Do subsampling
    int size_subsample = eta_subsample * (double) n;
    uvec indices_used = randperm(n, size_subsample);
    
    for(int i=0; i<size_subsample; i++){
      new_node->indices.push_back(indices_used(i));
    } 
  }else{
    //Don't subsample
    for(int i=0; i<n; i++){
      new_node->indices.push_back(i);
    } 
  }
  
  return new_node;
}

Node* class_boosting::get_new_node(Node* parent, bool this_is_left, int dim_selected, double location){
  
  //To the parent node, input the information on how this node is split
  //Input the dimension
  parent->dim_selected = dim_selected;
  //Input the partition point
  parent->location = location;
  
  double left = parent->left_points(dim_selected);
  double right = parent->right_points(dim_selected);
  
  parent->partition_point = left + location * (right - left);
  
  //Make a new child node
  Node* new_node = new Node;
  new_node->depth = parent->depth+1;

  new_node->left_points = parent->left_points;
  new_node->right_points = parent->right_points;
  
  if(this_is_left){
    new_node->node_id = 2 * parent->node_id;
    
    new_node->right_points(dim_selected) = parent->partition_point;
  }else{
    new_node->node_id = 2 * parent->node_id + 1;
    
    new_node->left_points(dim_selected) = parent->partition_point;
  }
  
  new_node->dim_selected = -1; //Not registered 
  new_node->location = 0; //Not registered 
  new_node->partition_point = 0; //Not registered 
  
  new_node->parent = parent;
  new_node->left = nullptr;
  new_node->right = nullptr;
  new_node->counts = 0;
  new_node->precision = compute_precision(new_node->depth);
  
  return new_node;
}


void class_boosting::add_children(Node* node, int dim_selected, double location){
  node->left = get_new_node(node, true, dim_selected, location);
  node->right = get_new_node(node, false, dim_selected, location);
}


void class_boosting::count_total_nodes( Node* node, int& count){
  
  //Make a stack for nodes
  std::stack<Node*> stack_tree;
  
  //current node
  Node* curr = node;
  
  while(curr != nullptr || stack_tree.empty() == false){
    
    while(curr != nullptr){
      stack_tree.push(curr);
      curr = curr->left;
    }
    
    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();

    curr = curr->right;
    
    ++count;
  }
  
}


Node* class_boosting::find_terminal_node(Node* root, vec& x){
  
  Node* curr = root;
  
  while(curr->left != nullptr){
    int dim_selected = curr->dim_selected;
    
    if(x(dim_selected) <= curr->partition_point){
      curr = curr->left;
    }else{
      curr = curr->right;
    }
    
  }
  
  return curr;
  
}
 
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//boosting functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


void class_boosting::boosting(){
  
  //tree boosting

  for(int index_tree=0; index_tree<num_trees; index_tree++){

    active_vars.fill(0);
    vars_chosen.fill(0);
    
    if(index_tree < num_each_dim * d){ //first stage
      int dim_curr = index_tree / num_each_dim;
      active_vars(dim_curr) = 1;
      
      is_first_stage = true;

    }else{//second stage
      active_vars.fill(1);
      
      is_first_stage = false;
    }
    
    Node* root = get_root_node();
    
    //construct a tree recursively
    d_store.clear();
    l_store.clear();
    theta_store.clear();
    
    construct_tree(root);
    
    //residualize
    vec x_temp;
    
    for(int i=0;i<n;i++){
      x_temp = residuals_current.col(i);
      residuals_current.col(i) = residualize(root, x_temp);
    } 
    

    //count # nodes
    int count = 0;
    count_total_nodes(root, count);
    tree_size_store(index_tree) = count;
    
    //check the maximum depth
    int depth_max = 0;
    check_max_depth(root, depth_max);
    max_depth_store(index_tree) = depth_max;
    
    //output the last residuals to check the performance
    if(index_tree == num_trees-1){
      residuals_last_boosting = residuals_current;
    }
    

    //summarize the information of the current tree in the list
    List list_curr_tree = Rcpp::List::create(Rcpp::Named("d") = d_store,
                                             Rcpp::Named("l") = l_store,
                                             Rcpp::Named("theta") = theta_store
    );
    
    tree_list.push_back(list_curr_tree);
    
    clear_node(root);
    
    
    print_progress_boosting(index_tree);

  }
  
}



void class_boosting::construct_tree(Node* node){

  //Make a stack for nodes
  std::stack<Node*> stack_tree;
  
  //current node
  Node* curr = node;
  
  while(curr != nullptr || stack_tree.empty() == false){
    
    while(curr != nullptr){
      stack_tree.push(curr);
      
      //decide whether or not to split the current node here
      
      bool is_split;
      
      //we stop splitting when we are at the bottom
      if(curr->depth > max_resol){
        is_split = false;
      }else{
        is_split = split_node(curr);
      }

      //store the information of the splitting rule chosen here
      if(is_split){
        d_store.push_back(curr->dim_selected);
        l_store.push_back(curr->location);
        theta_store.push_back(curr->theta);
      }else{
        d_store.push_back(-1);
        l_store.push_back(-1);
        theta_store.push_back(-1);
      }
      
      curr = curr->left;
    }
    
    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();
    
    curr = curr->right;
  }
  
}

//false: not split
//true: split
bool class_boosting::split_node(Node* node){
  
  //Make a matrix to compute the likelihood for possible splitting rule
  int n_current = node->indices.size();//#obs included in the current node
  
  bool is_split;
  
  if(n_current < min_obs){
    
    is_split = false;
    
  }else{
    //Compute the precision
    //double prec = get_precision(node) * (1 - learn_rate) / learn_rate * (double) n_current;
    double prec = get_precision(node) ;
    
    double alpha_l, alpha_r;
    double alpha_l_post, alpha_r_post;
    double n_left, n_right; //They are set to double to compute the likeihood
    double L_current;
    
    //what to do changes depending on in which stage we are
    
    for(int j=0; j<d; j++){
      
      if(active_vars(j) != 1){
        
        for(int i=0;i<num_grid_points_L;i++){
          log_like_matrix(j, i) = - LARGE_NUMBER;
        }
      }else{
        
        //if the width is too small, stop the splitting
        if(node->right_points(j) - node->left_points(j) < MIN_WIDTH){
          for(int i=0;i<num_grid_points_L;i++){
            log_like_matrix(j,i) = - LARGE_NUMBER;
          }
        }else{
          
          ivec left_counts = make_left_count_vector(node, j);
          
          for(int i=0;i<num_grid_points_L;i++){
            L_current = L_candidates(i);
            alpha_l = prec * L_current;
            alpha_r = prec * (1.0-L_current);
            
            n_left = (double) left_counts(i);
            n_right = (double) n_current - n_left;
            
            alpha_l_post = alpha_l + n_left;
            alpha_r_post = alpha_r + n_right;
            
            log_like_matrix(j, i) = log_beta(alpha_l_post, alpha_r_post) - log_beta(alpha_l, alpha_r)
              - n_left * log(L_current) - n_right * log(1.0-L_current);
          }
        }
        
      }

    }

    
    //Decide whether to divide the current node or node
    vec log_probs_split(2); //0: stop, 1: split
    
    
    double split_prob = get_split_prob(node);
    
    log_probs_split(0) = log(1-split_prob);
    log_probs_split(1) = log(split_prob) - log((double) sum(active_vars) * num_grid_points_L) + log_sum_mat(log_like_matrix);

    vec probs_split = log_normalize_vec(log_probs_split);
    
    if(R::runif(0, 1) < probs_split(0)){
      //Stop splitting
      
      //Don't forget clearn the current "indices" vector!
      //This is effective to save the memory cost
      node->indices.clear();
      
      is_split = false;
    }else{
      //Split
      //choose one splitting rule
      vec probs_rule;
      int rule_chosen;
      int dim_chosen;
      double location_chosen;
      
      double left_point;
      double right_point;
      double partition_point;
      
      probs_rule = log_normalize_vec(vectorise(log_like_matrix));
      
      rule_chosen = OneSample(probs_rule);
      dim_chosen = rule_chosen % d;
      location_chosen = L_candidates(rule_chosen / d);
      
      left_point = node->left_points(dim_chosen);
      right_point = node->right_points(dim_chosen);
      partition_point = left_point + (right_point - left_point) * location_chosen;
    
      node->dim_selected = dim_chosen;
      node->location = location_chosen;
      node->partition_point = partition_point;
      
      //allocate the observations into the two children nodes
      //At the same time, count them
      int n_l = 0, n_r = 0;

      node->left  = get_new_node(node, true, node->dim_selected, node->location);
      node->right = get_new_node(node, false, node->dim_selected, node->location);
      
      vector<int> indices_temp = node->indices;
      int size = indices_temp.size();
      for(int i=0; i<size; i++){
        int index_i = indices_temp[i];
        
        if(residuals_current(dim_chosen, index_i) < partition_point){
          node->left->indices.push_back(index_i);
          ++n_l;
        }else{
          node->right->indices.push_back(index_i);
          ++n_r;
        }
      }
      
      //Don't forget clean the current "indices" vector!
      //This is effective to save the memory cost
      node->indices.clear();
      
      double log_vol = sum(log(node->right_points - node->left_points));
      double learn_rate_modified = learn_rate * pow(1.0 - log_vol / log(2.0), - gamma);

      node->theta = (1 - learn_rate_modified) * location_chosen + learn_rate_modified * (double) n_l / (double) (n_l + n_r);

      //update the variable importance
      importances(dim_chosen) = importances(dim_chosen) + (double) n_l / (double) n * log(node->theta / location_chosen)
                                  + (double) n_r / (double) n * log((1-node->theta) / (1-location_chosen));
      
      is_split = true;
      
      //input the information of the selected variable
      //If the number of selected variable is equal to the threshold, we do not selec the other variables anymore
      vars_chosen(dim_chosen) = 1;
      
      if(sum(vars_chosen) == max_n_var){
        for(int j=0; j<d; j++){
          if(vars_chosen(j) == 0){
            active_vars(j) = 0;
          }
        }
      }
      
    }    
  }
  
  return is_split;
  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//utilities for boosting
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

double class_boosting::get_precision(Node* node){
  //Notice that the depth starts from 0
  return pow((double) node->depth+1, gamma);
}

//output: vector of observations that are "left" to each possible partition point
ivec class_boosting::make_left_count_vector(Node* node, int dim){
  //Make a vector of observations in the current dimension included in the current node
  vector<int> indices_temp = node->indices;
  int size = indices_temp.size();
  vec x_temp(size);
  for(int i=0; i<size; i++){
    x_temp(i) = residuals_current(dim, indices_temp[i]);
  }
  

  
  //Count values included in each interval
  ivec count_vec = ctree.make_count_vector(x_temp, node->left_points(dim), node->right_points(dim));
  
  return cumsum(count_vec);
}

double class_boosting::get_split_prob(Node* node){
  //Notice that the depth starts from 0
  return alpha * pow((double) node->depth+1, - beta);
}



void class_boosting::check_max_depth( Node* node, int& depth_max){
  
  //Make a stack for nodes
  std::stack<Node*> stack_tree;
  
  //current node
  Node* curr = node;
  
  while(curr != nullptr || stack_tree.empty() == false){
    
    while(curr != nullptr){
      stack_tree.push(curr);
      curr = curr->left;
    }
    
    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();
    
    //update the max depth here
    if(curr->depth > depth_max){
      depth_max = curr->depth;
    }
    
    curr = curr->right;
  }
  
}


void class_boosting::print_progress_boosting(int index_tree){
  
  if(num_trees < 30){
    Rcout << "boosting in progress..." << "\n";
  }else{
    if(index_tree == 0){
      Rcout << "boosting in progress..." << "\n";
      Rcout << "0" << "|--------------------------------------------------|" << "100%" << "\n";
      Rcout << " |" ;
    }else if(index_tree > 0){
      int r = num_trees/50;
      int rem = index_tree % r;
      
      if(rem == 0) {
        Rcout << "*" ;
      }
      
      if(index_tree == num_trees-1){
        Rcout << "|" << "\n";
      }
    }
  }
}



vec class_boosting::residualize( Node* root, vec& x){

  //find a terminal node that x belongs to
  Node* curr = find_terminal_node(root, x);
  
  //go up the tree one by one
  Node* parent;
  
  int dim_selected;
  
  double left_point;
  double right_point;
  double theta;
  
  vec resid_curr = x;
  
  while(curr->parent != nullptr){
    
    parent = curr->parent;
    
    dim_selected = parent->dim_selected;
    
    left_point = parent->left_points(dim_selected);
    right_point = parent->right_points(dim_selected);
    theta = parent->theta;
    
    resid_curr(dim_selected) = local_move(resid_curr(dim_selected), left_point, right_point, theta, parent->location, parent->left == curr);

    curr = parent;
  }

  return resid_curr;
}


double class_boosting::local_move(double x, double left_point, double right_point, 
                                       double theta, double area_ratio, bool left){
  
  if(left){
    return left_point + theta / area_ratio * (x - left_point);
  }else{
    return right_point + (1-theta) / (1-area_ratio) * (x - right_point);
  }
  
}

double class_boosting::evaluate_density(Node* root, vec& x){

  //find a terminal node that x belongs to
  Node* curr = find_terminal_node(root, x);
  
  //go up in the tree one by one
  Node* parent;
  
  double dens_curr = 0.0;
  
  while(curr->parent != nullptr){
    
    parent = curr->parent;
    
    if(parent->left == curr){
      dens_curr  = dens_curr + log(parent->theta) - log(parent->location);
    }else{
      dens_curr  = dens_curr + log(1-parent->theta) - log(1-parent->location);
    }
    
    curr = parent;
  }
  
  return dens_curr;
  
}

double class_boosting::evaluate_log_prior(Node* node){
  if(node->left == nullptr){
    return log( 1.0 - compute_splitting_prob(node->depth) );
  }else{
    double log_prob_divide = log(compute_splitting_prob(node->depth));
    double log_dens_theta = R::dbeta( node->theta, 
                                  node->precision * node->location,
                                  node->precision * (1.0 - node->location),
                                  true);
    
    log_dens_theta = 0;

    return log_prob_divide + log_dens_theta + evaluate_log_prior(node->left) + evaluate_log_prior(node->right);
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//make outputs
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

List class_boosting::output(mat support_store){
  
  List out;
  
  out = Rcpp::List::create(     Rcpp::Named("residuals_boosting") = residuals_last_boosting,
                                Rcpp::Named("tree_size_store") = tree_size_store,
                                Rcpp::Named("max_depth_store") = max_depth_store,
                                Rcpp::Named("variable_importance") = importances,
                                Rcpp::Named("support") = support_store,
                                Rcpp::Named("tree_list") = tree_list
                                );

  
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//destructor
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

class_boosting::~class_boosting(){
  //do nothing in the current version
}


void class_boosting::clear_node(Node* root){

  //Make a stack for nodes
  std::stack<Node*> stack_tree;
  
  //current node
  Node* curr = root;
  
  while(curr != nullptr || stack_tree.empty() == false){
    
    while(curr != nullptr){
      stack_tree.push(curr);
      curr = curr->left;
    }
    
    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();
    
    Node* curr_old = curr;
    Node* curr_new = curr->right;
    
    delete curr_old;
    
    //Going up one by one, find a right node that is "alive"
    curr = curr_new;
  }
  
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//miscellaneous functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////


double class_boosting::compute_precision( int depth){
  return  precision;
}

double class_boosting::compute_splitting_prob(int depth){
  return  alpha * pow((double) depth + 1.0, - beta);
}
