boosting = function( data,
                     add_noise = TRUE,
                     margin_size = 1.0,
                     ntree_marginal = 100,
                     ntree_dependence = 1000,
                     c0 = 0.1,
                     gamma = 0.1,
                     max_resol = 50,
                     min_obs = 5,
                     eta_subsample = 1.0,
                     alpha = 0.5,
                     beta = 0.0,
                     precision = 1.0,
                     J = 8,
                     max_n_var = 100
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
  
  start_time = Sys.time()
  
  #do boosting
  out = do_boosting(X_new,
                    precision,
                    alpha,
                    beta,
                    gamma,
                    max_resol,
                    ntree_marginal,
                    ntree_dependence,
                    c0,
                    min_obs,
                    J,
                    margin_size,
                    eta_subsample,
                    max_n_var
  )
  
  end_time = Sys.time()
  
  print(end_time - start_time)
  
  return(out)
}

simulation_b = function(list_boosting, size){
  return(simulation(list_boosting$tree_list, size, list_boosting$support))
}

eval_density_b = function(list_boosting, eval_points){
  out_dens = evaluate_log_density(list_boosting$tree_list, eval_points, list_boosting$support)
  out = list(out_dens$log_densities, out_dens$mean_log_dens_path)
  names(out) = c("log_densities", "mean_log_dens_path")
  return(out)
}