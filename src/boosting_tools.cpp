// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

bool Is_included(const vec x, const vec vec_l, const vec vec_r){
  bool out = all(vec_l < x) && all(x <= vec_r);
  return out;
}

ivec Vec_of_is_included(const int N, const mat& Z,const vec vec_l,const vec vec_r){
  ivec is_included(N);

  for(int i=0;i<N;i++){
    vec z_i = Z.row(i).t();

    if(Is_included(z_i, vec_l, vec_r) == 1){
      is_included(i) = 1;
    }else{
      is_included(i) = 0;
    }
  }

  return is_included;
}

double Move_point(double z, double a, double b, double c, double theta_post, int is_left){

  double G_A;

  if(is_left == 1){
    G_A = theta_post * (z-a) / (c-a);
  }else{
    G_A = theta_post + (1-theta_post) * (z-c) / (b-c);
  }

  double out = a + (b-a) * G_A;

  return out;
}


// [[Rcpp::export]]
List residualize(mat X, mat grid_points, mat tree_l0, mat tree_r0, ivec levels, mat post_states0, ivec Is_non_terminal, imat children_IDs, double c_learn){
  //Input the information
  int n = X.n_rows;
  int d = X.n_cols;
  //int I = post_states0.n_cols;

  int n_grid_points = grid_points.n_rows;
  int n_nodes = tree_l0.n_rows;

  int n_division = (n_nodes-1) / 2;

  //Sort the nodes so that the bottom nodes come first and the root node comes the last
  mat tree_l(n_nodes, d);
  mat tree_r(n_nodes, d);
  //mat post_states(n_division, I);
  int max_K = arma::max(levels);
  int index_node = 0;
  int index_node_par = 0;

  ivec IDs_prev(n_nodes);

  for(int k=max_K;k>-1;k--){
    uvec indices_k = find(levels == k);
    int n_nodes_k = indices_k.n_rows;

    for(int j=0;j<n_nodes_k;j++){
      uword index_j = indices_k(j);
      tree_l.row(index_node) = tree_l0.row(index_j);
      tree_r.row(index_node) = tree_r0.row(index_j);

      IDs_prev(index_node) = index_j;
      index_node = index_node + 1;

      if(Is_non_terminal(index_j) == 1){
        //post_states.row(index_node_par) = post_states0.row(index_j);
        index_node_par = index_node_par + 1;
      }
    }
  }

  //Move the data points in the bottom-up manner
  mat X_new = X;
  mat grid_points_new = grid_points;

  //At the same time, compute the density at each grid point
  vec densities(n_grid_points);
  densities.fill(1.0);

  //To compute the variable importance, store the information of theta and the KL on each node
  vec theta_post_store(n_nodes);
  theta_post_store.fill(-1.0);
  vec KL_store(n_nodes);
  KL_store.fill(-1.0);

  ivec dim_divide_store(n_nodes);

  for(int j=0;j<n_division;j++){
    //Get the information of the current children nodes
    //At each iteration, we pick up 2 nodes. (The root node is not touched.)
    vec left_child_l  = tree_l.row(2*j).t();
    vec left_child_r  = tree_r.row(2*j).t();

    vec right_child_l = tree_l.row(2*j+1).t();
    vec right_child_r = tree_r.row(2*j+1).t();

    uvec temp = find(left_child_l != right_child_l);
    int dim_divide = (int) temp(0);

    double a = left_child_l(dim_divide);
    double b = right_child_r(dim_divide);
    double c = left_child_r(dim_divide);

    //First, move the data points
    ivec is_included_X_l = Vec_of_is_included(n, X_new, left_child_l, left_child_r);
    ivec is_included_X_r = Vec_of_is_included(n, X_new,right_child_l,right_child_r);

    uvec indices_X_l = find(is_included_X_l == 1);
    uvec indices_X_r = find(is_included_X_r == 1);

    int n_l = sum(is_included_X_l);
    int n_r = sum(is_included_X_r);
    int n = n_l + n_r;

    //Compute the posterior mean of theta.
    double theta_post;

    if(n == 0){
      theta_post = (c-a) / (b-a);
    }else{
      theta_post = (1-c_learn) * (c-a) / (b-a) + c_learn * n_l / (n_l + n_r);
    }

    //Store the information of the theta_post and KL divergence.
    theta_post_store(IDs_prev(2*j)) = theta_post;
    theta_post_store(IDs_prev(2*j+1)) = theta_post;

    double KL_temp = theta_post * (log(theta_post) - log( (c-a) / (b-a) )) + (1-theta_post) * (log(1-theta_post) - log( (b-c) / (b-a) ));

    KL_store(IDs_prev(2*j)) = KL_temp;
    KL_store(IDs_prev(2*j+1)) = KL_temp;

    dim_divide_store(IDs_prev(2*j)) = dim_divide;
    dim_divide_store(IDs_prev(2*j+1)) = dim_divide;

    //Update the data points
    //left
    for(int i=0;i<n_l;i++){
      uword index_i = indices_X_l(i);
      X_new(index_i,dim_divide) = Move_point(X_new(index_i,dim_divide), a, b, c, theta_post, 1);
    }

    //right
    for(int i=0;i<n_r;i++){
      uword index_i = indices_X_r(i);
      X_new(index_i,dim_divide) = Move_point(X_new(index_i,dim_divide), a, b, c, theta_post, 0);
    }

    //Second, move the grid points
    //At the same time, update the densities
    ivec is_included_gp_l = Vec_of_is_included(n_grid_points, grid_points_new, left_child_l, left_child_r);
    ivec is_included_gp_r = Vec_of_is_included(n_grid_points, grid_points_new,right_child_l,right_child_r);

    uvec indices_gp_l = find(is_included_gp_l == 1);
    uvec indices_gp_r = find(is_included_gp_r == 1);

    //Update the grid points
    //left
    for(int i=0;i<(int)indices_gp_l.n_rows;i++){
      uword index_i = indices_gp_l(i);
      grid_points_new(index_i,dim_divide) = Move_point(grid_points_new(index_i,dim_divide), a, b, c, theta_post, 1);

      double dens_temp = densities(index_i);
      densities(index_i) = dens_temp * theta_post / ( (c-a)/(b-a) );
    }

    //right
    for(int i=0;i<(int)indices_gp_r.n_rows;i++){
      uword index_i = indices_gp_r(i);
      grid_points_new(index_i,dim_divide) = Move_point(grid_points_new(index_i,dim_divide), a, b, c, theta_post, 0);

      double dens_temp = densities(index_i);
      densities(index_i) = dens_temp * (1-theta_post) / ( (b-c)/(b-a) );
    }

  }

  //Next we compute the variable importance
  vec importance(d);
  importance.fill(0);
  vec measure_post(n_nodes);
  measure_post(0) = 1.0;

  for(int i=0;i<n_nodes;i++){
    if(Is_non_terminal(i) == 1){
      int child_ID_temp = children_IDs(0,i);
      double theta_temp = theta_post_store(child_ID_temp);

      measure_post(child_ID_temp) = measure_post(i) * theta_temp;
      measure_post(child_ID_temp+1) = measure_post(i) * (1-theta_temp);

      int dim_divide_temp = dim_divide_store(child_ID_temp);

      importance(dim_divide_temp) = importance(dim_divide_temp) + measure_post(i) * KL_store(child_ID_temp);
    }
  }

  List out = Rcpp::List::create(Rcpp::Named("X_new") = X_new,
                                Rcpp::Named("grid_points_new") = grid_points_new,
                                Rcpp::Named("pred_densities") = densities,
                                Rcpp::Named("importance") = importance,
                                Rcpp::Named("theta_post") = theta_post_store);

  return out;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double Move_point_inv(const double y,const double a,const double b,const double c,const double theta_post,const int is_left){

  double z = (y-a)/(b-a);
  double out;

  if(is_left == 1){
    out = a + (c-a)/theta_post * z;
  }else{
    out = c + (b-c)/(1-theta_post) * (z - theta_post);
  }

  return out;
}

// [[Rcpp::export]]
mat G_inverse(mat Y, mat tree_l, mat tree_r, vec theta_post, mat children_IDs){
  //Input the basic information
  int n = Y.n_rows;
  int n_nodes = tree_l.n_rows;

  mat Y_new = Y;

  //Move the input points from the root node to the bottom nodes.
  for(int j=0;j<n_nodes;j++){
    int child_ID_l = children_IDs(0,j);
    
    //Rcout << child_ID_l << "\n";

    //If the current node is not a leaf, we move the points included in the node
    if(child_ID_l != -1){
      vec left = tree_l.row(j).t();
      vec right = tree_r.row(j).t();

      //Find the points included in the current node
      ivec is_included_Y = Vec_of_is_included(n, Y_new, left, right);
      uvec indices_Y = find(is_included_Y == 1);

      //theta = G(Al|A)
      double theta_current = theta_post(child_ID_l);

      //Get the information of the children nodes
      vec left_child_l  = tree_l.row(child_ID_l).t();
      vec left_child_r  = tree_r.row(child_ID_l).t();

      vec right_child_l = tree_l.row(child_ID_l+1).t();
      vec right_child_r = tree_r.row(child_ID_l+1).t();

      uvec temp = find(left_child_l != right_child_l);
      int dim_divide = (int) temp(0);

      double a = left_child_l(dim_divide);
      double b = right_child_r(dim_divide);
      double c = left_child_r(dim_divide);

      //The next operation changes dependind on whether the value in the focused dimension is less than the value "mid" or not
      double mid = a + theta_current * (b-a);
      mat Y_current = Y_new.rows(indices_Y);
      vec Y_current_marginal = Y_current.col(dim_divide);

      uvec indices_Y_l0 = find(Y_current_marginal < mid);
      uvec indices_Y_r0 = find(mid < Y_current_marginal);

      uvec indices_Y_l = indices_Y(indices_Y_l0);
      uvec indices_Y_r = indices_Y(indices_Y_r0);

      int n_l = indices_Y_l.size();
      int n_r = indices_Y_r.size();

      //Update the grid points
      //left
      for(int i=0;i<(int)n_l;i++){
        uword index_i = indices_Y_l(i);
        Y_new(index_i,dim_divide) = Move_point_inv(Y_new(index_i,dim_divide), a, b, c, theta_current, 1);
      }

      //right
      for(int i=0;i<(int)n_r;i++){
        uword index_i = indices_Y_r(i);
        Y_new(index_i,dim_divide) = Move_point_inv(Y_new(index_i,dim_divide), a, b, c, theta_current, 0);
      }

    }
  }

  return Y_new;
}
