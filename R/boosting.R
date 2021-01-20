boosting = function(data, #Data matrix
                   grid.points, #Grid points used to compute the predictive density
                   max.resol = 3, #Maximum resolution
                   n.particle = 100, #Number of particles
                   n.grid.L = 31, #Number of grid points for the L's prior
                   c = 0.1, #Lerning rate
                   n.trees = 50, #The number of trees
                   out.pred.scores = FALSE, #If TRUE, output the predictive scores
                   out.for.simulation = FALSE #If TRUE, output the results necessary for simulation
){
  if(ncol(data) == ncol(grid.points)){
    d = ncol(data)
    n = nrow(data)
    n_grid = nrow(grid.points)
  }else{
    print("Error: The dimension of the data set and the grid points do not match")
    return(0)
  }
  
  X_current = data
  grid_points_current = grid.points
  density_current = rep(1,nrow(grid_points_current))
  importance_store = numeric(d)
  pred_score_store = numeric(n.trees)
  
  if(out.for.simulation == TRUE){
    result_store = list()
  }

  #Implement the FS algorithm
  for(index_tree in 1:n.trees){
    
    #SMC
    model_parameters_list = list(c)
    names(model_parameters_list) = c("c")
    
    out = SMCforPT(data,
                   1,
                   rep(1, n),
                   grid.points,
                   rep(1,n_grid),
                   0.1,
                   1,
                   model_parameters_list,
                   n.particle,
                   max.resol,
                   n.grid.L+1,
                   0.5,
                   0.1,
                   5,
                   1,
                   3
    )
    

    tree_left = t(out$tree_left)
    tree_right = t(out$tree_right)
    n_nodes = nrow(tree_left)
    levels = numeric(n_nodes)
    post_states = t(out$posterior_states)
    
    current_level = 0
    for(i in 1:n_nodes){
      levels[i] = current_level
      if(prod(tree_right[i,] == rep(1,d))){
        current_level = current_level + 1
      }
    }
    
    children_IDs = out$children_IDs
    Is_non_terminal = numeric(n_nodes)
    Is_non_terminal[which(children_IDs[1,] > 1)] = 1
    
    #Residualization + Computing the variable iportance
    out_normalize = residualize(X_current, grid_points_current, tree_left, tree_right, levels, post_states, Is_non_terminal, children_IDs, c)

    importance_store = importance_store + out_normalize$importance
    density_current = density_current * out_normalize$pred_densities
    
    pred_score_store[index_tree] = mean(log(density_current + 1e-100))
    
    if(out.for.simulation==TRUE){
       result_store[[index_tree]] = list("tree_left" = tree_left,
                                        "tree_right" = tree_right,
                                        "children_IDs" = children_IDs+1,
                                        "theta_post" = out_normalize$theta_post)
    }
    
    if(index_tree < n.trees){
      X_current = out_normalize$X_new
      grid_points_current = out_normalize$grid_points_new
    }
    
    #print(paste("# trees", index_tree))
  }
  
  out_boosting = list("densities" = density_current,
                      "variable.importance" = importance_store) 
  
  if(out.pred.scores == TRUE){
    out_boosting[["pred.scores"]] = pred_score_store
  }
  
  if(out.for.simulation==TRUE){
    out_boosting[["result.for.simulation"]] = result_store
  }
  
  return(out_boosting)
}


simulation = function(result.for.simulation, #The information of the measures obtained in the FS algorithm
                      N #The size of the simulated data
                      ){
  
  n.trees = length(result.for.simulation)
  simulation.current = matrix(runif(d*N),nrow=N,ncol=d)
  
  for(index_tree in n.trees:1){
    result_current = result.for.simulation[[index_tree]]
    tree_left = result_current$tree_left
    tree_right = result_current$tree_right
    children_IDs = result_current$children_IDs-1
    theta_post = result_current$theta_post
    
    simulation.current = G_inverse(simulation.current, tree_left, tree_right, theta_post, children_IDs)
  }
  
  return(simulation.current)
}