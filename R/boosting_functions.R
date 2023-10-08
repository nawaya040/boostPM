boosting = function(data, # data: d x n matrix
                   add_noise = TRUE, # add noise if we want to avoid duplicated values 
                   Omega = NULL, # sample space: automatically created when NULL
                   ntree_max_marginal = 100, # maximum # trees used for marginal distributions
                   ntree_max_dependence = 1000, # maximum # trees used for dependence structures
                   c0 = 0.1, # global shrinkage parameter
                   gamma = 0.1, # local shrinkage parameter
                   max_resol = 15, # maximum resolution
                   min_obs = 5, # minimum # observations allowed to included in a single node
                   early_stop = NULL, # parameter for early stopping (may need to be modified)
                   alpha = 0.9, # prior prob of dividing a node = alpha * (1 + depth)^beta
                   beta = 0.0, 
                   precision = 1.0, # precision of the beta prior
                   nbins = 8, # # bins. we check nbins-1 possible partition points 
                   max_n_var = 100 # this is an experimental one so should be set to d 
                     ){

  
  #preprocess the data if necessary
  if(add_noise){
    
    d = ncol(data)
    n = nrow(data)
    
    X_new = matrix(NA, nrow = n, ncol = d)
    
    for(j in 1:d){
      sort_temp = sort(data[,j], index.return = T)
      Xj = sort_temp$x
      indices_unique = which(!duplicated(Xj))
      num_unique = length(indices_unique)
      
      add_noise = function(k){
        
        ind = indices_unique[k]
        if(k < num_unique){
          ind_last = indices_unique[k+1] - 1
        }else{
          ind_last = n
        }
        
        if(ind != ind_last){
          #Ties found
          #determine the size of noise based on adjacent values
          if(k == 1){
            left = right = (Xj[indices_unique[2]] - Xj[indices_unique[1]]) / 2
          }else if(k == num_unique){
            left = right = (Xj[indices_unique[num_unique]] - Xj[indices_unique[num_unique-1]]) / 2
          }else{
            left  = (Xj[indices_unique[k]]   - Xj[indices_unique[k-1]]) / 2
            right = (Xj[indices_unique[k+1]] - Xj[indices_unique[k]]  ) / 2
          }
          
          #add noise to avoid duplication
          return(Xj[ind:ind_last] + runif(ind_last-ind+1, -left, right))
        }else{
          #No ties
          #output the original value
          return(Xj[ind])
        }
        
      }
      
      X_new[sort_temp$ix, j] = unlist(sapply(1:num_unique, add_noise))
      
    }
    
  }else{
    X_new = data
  }
  
  # make a matrix that indicates "Omega" (= the sample space) 
  # if this is already input by the user, no need to make it
  # if not, make Omega based on the rage
  
  if(is.null(Omega)){
    
    d = ncol(data)
    Omega = matrix(NA, nrow = d, ncol = 2)
    
    for(j in 1:d){
      min_j = min(X_new[,j]);
      max_j = max(X_new[,j]);
      width_j = max_j - min_j;
      
      m_resize = min_j - 0.1 * width_j;
      M_resize = max_j + 0.1 * width_j;
      
      Omega[j,1] = m_resize;
      Omega[j,2] = M_resize;
      
      X_new[,j] = (X_new[,j] - m_resize) / (M_resize - m_resize)
    }
  }else{
    d = ncol(data)
    
    # check if all observations are included in the sample space specified by the user
    for(j in 1:d){
      is_okay = prod((Omega[j,1] < data[,j])) * prod((Omega[j,2] > data[,j]))
      if(is_okay != 1){
        stop("The sample space (omega) is too small and some observations are outside")
      }
    }

    for(j in 1:d){
      X_new[,j] = (X_new[,j] - Omega[j,1]) / (Omega[j,2] - Omega[j,1])
    }
    
  }
  
  if(is.null(early_stop)){
    eta_subsample = 1.0
    # the following numbers are random
    # just to avoid errors
    thresh_stop = 1.0
    ntrees_wait = 100 
  }else{
    eta_subsample = 0.9
    thresh_stop = early_stop[1]
    ntrees_wait = early_stop[2]
  }
  
  start_time = Sys.time()
  
  #do boosting
  out = do_boosting(X_new,
                    precision,
                    alpha,
                    beta,
                    gamma,
                    max_resol,
                    ntree_max_marginal,
                    ntree_max_dependence,
                    c0,
                    min_obs,
                    nbins,
                    eta_subsample,
                    thresh_stop,
                    ntrees_wait,
                    max_n_var
  )
  
  out$Omega = Omega
  
  end_time = Sys.time()
  out$time = end_time - start_time
  
  print(end_time - start_time)
  
  return(out)
}

simulation_b = function(list_boosting, size){
  return(simulation(list_boosting$tree_list, size, list_boosting$Omega))
}

eval_density_b = function(list_boosting, eval_points){
  out_dens = evaluate_log_density(list_boosting$tree_list, eval_points, list_boosting$Omega)
  out = list(out_dens$log_densities, out_dens$mean_log_dens_path)
  names(out) = c("log_densities", "mean_log_dens_path")
  return(out)
}