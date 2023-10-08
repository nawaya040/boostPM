#ifndef post_H
#define post_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

struct Node
{
  vec left_points;
  vec right_points;
  
  int dim_selected; 
  
  double location; // the value of "L"
  double partition_point; //The partition point in the selected dimension
  
  double theta;
  
  Node* parent = nullptr;
  Node* left = nullptr;
  Node* right = nullptr;
};

//Main functions
mat simulation(List tree_list, int size_simulation, mat support);
vec update_vec(Node* node, vec& x);
List evaluate_log_density(List tree_list, mat eval_points, mat support);

//Tree functions

Node* get_root_node();
Node* get_new_node(Node* parent, bool this_is_left, int dim_selected, double location);

void construct_tree(Node* node, List tree_current);

double evaluate_density(Node* root, vec& x);
vec residualize( Node* root, vec& x);

Node* find_terminal_node(Node* root, vec& x);

double local_move(double x, double left_point, double right_point, 
                  double theta, double area_ratio, bool left);
  
//for cleaning
void clear_node(Node* root);

#endif