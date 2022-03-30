#ifndef CTREE_H
#define CTREE_H

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

struct cNode
{
  unsigned int depth;
  
  double left_point;
  double right_point;
  double middle_point;
  
  unsigned int counts = 0;
  
  cNode* left = nullptr;
  cNode* right = nullptr;
};

class count_tree{
public:
  
  //variables
  int J;
  int n_cells;
  cNode* root;
  int index_bottom_nodes;
  ivec out_vec;
  
  //treefunctions
  void init(int J_input);
  cNode* get_new_cnode(int depth);
  cNode* construct_tree(cNode* cnode);
  void update_intervals(cNode* cnode, double left_point, double right_point);
  void initialize_counts(cNode* cnode);
  void add_one(cNode* cnode, double z);
  void pickup_bottom_counts(cNode* cnode);
  
  //function to input the information and output the count vector
  ivec make_count_vector(vec x, double left_point, double right_point);
  
  //destructor 
  ~count_tree();
  void clear_cnode(cNode* root);
  
};

#endif