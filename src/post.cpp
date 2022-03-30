// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include<bits/stdc++.h>
#include "post.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

int d_g;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// [[Rcpp::export]]
mat simulation(List tree_list, int size_simulation, mat support){
  
  //initialization
  d_g = support.n_rows;
  mat current_mat = zeros(d_g, size_simulation);
  
  for(int i=0; i<size_simulation; i++){
    vec temp = Rcpp::runif(d_g, 0.0, 1.0);
    current_mat.col(i) = temp;
  }
  
  int num_trees = tree_list.size();

  for(int index_tree = num_trees-1; index_tree>-1; index_tree--){
    //reconstruct a tree
    Node* root = get_root_node();
    construct_tree(root, tree_list[index_tree]);
    
    //update the current values with the top-down algorithm
    vec x_temp;
    for(int i=0; i<size_simulation; i++){
      x_temp = current_mat.col(i);
      current_mat.col(i) = update_vec(root, x_temp);
    }

    //delete the current tree
    clear_node(root);
  }
  
  for(int i=0; i<size_simulation; i++){
    vec temp = current_mat.col(i);
    for(int j=0; j < d_g; j++){
      temp(j) = support(j,0) + temp(j) * (support(j,1) - support(j,0));
    }

    current_mat.col(i) = temp;
  }
  
  return current_mat.t();
}


vec update_vec(Node* node, vec& x){
  vec x_new = x;
  
  Node* curr = node;
  
  int dim;
  double a, b, c;
  double theta;
  double thresh;
  double y, z;
  
  while(curr->left != nullptr){
    
    //get the information of the current node
    dim = curr->dim_selected;
    a = curr->left_points(dim);
    b = curr->right_points(dim);
    c = curr->partition_point;
    theta = curr->theta;
    
    //move a value in the chosen dimension
    thresh = a + theta * (b - a);
    y = x_new(dim);
    z = (y - a) / (b - a);

    if(y < thresh){
      x_new(dim) = a + (c-a) / theta * z;
      
      curr = curr->left;
      
    }else{
      x_new(dim) = c + (b-c) / (1-theta) * (z - theta);
      
      curr = curr->right;
    }
    
  }
  
  return x_new;
}


// [[Rcpp::export]]
List evaluate_log_density(List tree_list, mat eval_points, mat support){
  //initialization
  d_g = support.n_rows;
  
  vec log_width_store = zeros(d_g);
  
  for(int j=0; j<d_g; j++){
    double m_resize = support(j,0);
    double M_resize = support(j,1);
    
    eval_points.col(j) = (eval_points.col(j) - m_resize) / (M_resize - m_resize);
    
    log_width_store(j) = log(M_resize - m_resize);
  }
  
  double sum_log_width = sum(log_width_store);
  
  int n_eval = eval_points.n_rows;
  mat residuals_eval_points = eval_points.t();
  
  vec log_densities_boosting = zeros(n_eval);
  
  int num_trees = tree_list.size();
  
  vec mean_log_dens_path = zeros(num_trees);
  
  for(int index_tree = 0; index_tree<num_trees; index_tree++){
    //reconstruct a tree
    Node* root = get_root_node();
    construct_tree(root, tree_list[index_tree]);
    
    //evaluate densities
    vec x_temp;
    for(int i=0; i<n_eval; ++i){
      x_temp = residuals_eval_points.col(i);
      log_densities_boosting(i) = log_densities_boosting(i) + evaluate_density(root, x_temp);
    }
    
    for(int i=0;i<n_eval;i++){
      x_temp = residuals_eval_points.col(i);
      residuals_eval_points.col(i) = residualize(root, x_temp);
    } 
    
    //record progress
    mean_log_dens_path(index_tree) = mean(log_densities_boosting) - sum_log_width;
    
    //delete the current tree
    clear_node(root);
  }
  
  List out;
  
  out = Rcpp::List::create(Rcpp::Named("log_densities") = log_densities_boosting - sum_log_width,
                                Rcpp::Named("mean_log_dens_path") = mean_log_dens_path);
  
  return out;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tree functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Node* get_root_node(){
  Node* new_node = new Node;
  
  new_node->left_points = zeros<vec>(d_g);
  new_node->right_points = ones<vec>(d_g);
  
  new_node->dim_selected = 0; //Not registered 
  new_node->location = 0; //Not registered 
  new_node->partition_point = 0; //Not registered 
  
  new_node->parent = nullptr;
  new_node->left = nullptr;
  new_node->right = nullptr;

  return new_node;
}

Node* get_new_node(Node* parent, bool this_is_left, int dim_selected, double location){

  double left = parent->left_points(dim_selected);
  double right = parent->right_points(dim_selected);
  
  parent->partition_point = left + location * (right - left);
  
  //Make a new child node
  Node* new_node = new Node;

  new_node->left_points = parent->left_points;
  new_node->right_points = parent->right_points;
  
  if(this_is_left){
    new_node->right_points(dim_selected) = parent->partition_point;
  }else{
    new_node->left_points(dim_selected) = parent->partition_point;
  }
  

  new_node->dim_selected = 0; //Not registered 
  new_node->location = 0; //Not registered 
  new_node->partition_point = 0; //Not registered 
  
  new_node->parent = parent;
  new_node->left = nullptr;
  new_node->right = nullptr;

  return new_node;
}


void construct_tree(Node* node, List tree_current){
  
  ivec d_store = tree_current["d"];
  vec l_store = tree_current["l"];
  vec theta_store = tree_current["theta"];
  
  //Make a stack for nodes
  std::stack<Node*> stack_tree;
  
  //current node
  Node* curr = node;
  int index_node = 0;
  
  while(curr != nullptr || stack_tree.empty() == false){
    
    while(curr != nullptr){
      stack_tree.push(curr);
      
      //check whether or not to split the current node here
      
      bool is_split = (d_store(index_node) > -1);
      
      //split the current node if necessary
      if(is_split){
        curr->dim_selected = d_store(index_node);
        curr->location = l_store(index_node);
        curr->theta = theta_store(index_node);
        
        curr->left = get_new_node(curr, true, curr->dim_selected, curr->location);
        curr->right = get_new_node(curr, false, curr->dim_selected, curr->location);
      }
      
      index_node++;
      
      curr = curr->left;
    }
    
    //the current node must be nullptr at this point
    curr = stack_tree.top();
    stack_tree.pop();
    
    curr = curr->right;
  }

}

double evaluate_density(Node* root, vec& x){
  
  //finde a terminal node that x belongs to
  Node* curr = find_terminal_node(root, x);
  
  //go up the tree one by one
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

vec residualize( Node* root, vec& x){
  
  //finde a terminal node that x belongs to
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

Node* find_terminal_node(Node* root, vec& x){
  
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

double local_move(double x, double left_point, double right_point, 
                   double theta, double area_ratio, bool left){
  
  if(left){
    return left_point + theta / area_ratio * (x - left_point);
  }else{
    return right_point + (1-theta) / (1-area_ratio) * (x - right_point);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// for clearning
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void clear_node(Node* root){
  
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