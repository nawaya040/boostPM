// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "count_tree.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

#define ZERO_CONST 0

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//initialization
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void count_tree::init(int J_input){
  //input J
  J = J_input;
  
  //# total cells
  n_cells = pow(2, J);
  
  //construct the tree
  root = get_new_cnode(ZERO_CONST);
  root = construct_tree(root);
  
  //Make a vector to output
  out_vec = zeros<ivec>(n_cells);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//ctree functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

cNode* count_tree::get_new_cnode(int depth){
  cNode* new_cnode = new cNode;
  
  new_cnode->depth = depth;
  
  return new_cnode;
}

cNode* count_tree::construct_tree(cNode* cnode){
  
  int depth = cnode->depth;
  
  if(depth < J){
    cNode* cnode_l = get_new_cnode(depth+1);
    cnode->left = construct_tree(cnode_l);
    
    cNode* cnode_r = get_new_cnode(depth+1);
    cnode->right = construct_tree(cnode_r);
  }
  
  return cnode;
}

void count_tree::update_intervals(cNode* cnode, double left_point, double right_point){
  cnode->left_point = left_point;
  cnode->right_point = right_point;
  
  double middle_point = (left_point + right_point) * 0.5;
  
  cnode->middle_point = middle_point;
  
  if(cnode->left != nullptr){
    update_intervals(cnode->left,  left_point,   middle_point);
    update_intervals(cnode->right, middle_point, right_point);
  }
}

void count_tree::initialize_counts(cNode* cnode){
  cnode->counts = 0;
  if(cnode->left != nullptr){
    initialize_counts(cnode->left);
    initialize_counts(cnode->right);
  }
}

void count_tree::add_one(cNode* cnode, double z){
  ++cnode->counts;
  
  if(cnode->left != nullptr){
    if(z < cnode->middle_point){
      add_one(cnode->left, z);
    }else{
      add_one(cnode->right, z);
    }
  }
}

void count_tree::pickup_bottom_counts(cNode* cnode){
  if(cnode->left == nullptr){
    out_vec(index_bottom_nodes) = cnode->counts;
    ++index_bottom_nodes;
  }else{
    pickup_bottom_counts(cnode->left);
    pickup_bottom_counts(cnode->right);
  }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//function to input the information and output the count vector
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

ivec count_tree::make_count_vector(vec x, double left_point, double right_point){
  //update intervals
  update_intervals(root, left_point, right_point);
  
  //update counts 
  initialize_counts(root);
  
  int N = x.n_rows;
  for(int i=0; i<N; i++){
    add_one(root, x(i));
  }
  
  //Make an out based on the terminal nodes
  index_bottom_nodes = 0;
  pickup_bottom_counts(root);
  
  return out_vec;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
//destructor
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

count_tree::~count_tree(){
  clear_cnode(root);
}

void count_tree::clear_cnode(cNode* root){
  if(root){
    clear_cnode(root->left);
    clear_cnode(root->right);
    delete root;
  }
}