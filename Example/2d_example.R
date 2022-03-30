#Generate data from a beta mixture
set.seed(11)
n = 5000
d=2
data = matrix(NA, nrow=n, ncol=d)

for(i in 1:n){
  u = runif(1)
  if(u < 0.6){
    for(k in 1:2){
      data[i,k] = rbeta(1,40,80)
    }
  }else{
    for(k in 1:2){
      data[i,k] = rbeta(1,90,30)
    }
  }
}

plot(data[,1], data[,2], xlab = "x1", ylab = "x2", xlim=c(0,1), ylim=c(0,1), main = "observations")

#Make the grid points
x.grid = seq(0.005,0.995,by=0.01)
y.grid = seq(0.005,0.995,by=0.01)
grid.points = as.matrix(expand.grid(x.grid,y.grid))

#Run boosting
out = boosting(# parameters for boosting
               data = data, #data = n X d matrix
               add_noise = FALSE, # add uniform noises if there are tied values
               margin_size = 1.0, # size of margin we make when scaling data into (0,1]
                                  # if data is defined in (0,1]^d, we can set it to 0.0 = no scaling 
               ntree_marginal = 100, # # trees per dimension used in the first stage
               ntree_dependence = 1000, # # trees used in the second stage
               c0 = 0.1, # c0 = global scale of the learning parameter
               gamma = 0.1, # gamma = stronger regularization for small nodes
               max_resol = 50, # maximum resolution (depth) of trees
               min_obs = 5, # if # obs in a node > min_obs, this node is no longer split
               eta_subsample = 1.0, # eta * n observations are used to learn a new tree
               J = 8, # # grid points for splitting = 2^J-1 
               max_n_var = d, # this is an experimental one so should be set to d 
               # parameters for the PT-based weak learner
               alpha = 0.5, # prior prob of dividing a node = alpha * (1 + depth)^beta
               beta = 0.0,
               precision = 1.0 # precision of the beta prior
)

#Evaluate the log-densities
out_dens = eval_density_b(list_boosting = out, # simply use the output of the boosting function
                          eval_points = grid.points # matrix of evaluation points
                          )

#Visualize the estimated density function
library(ggplot2)
library(viridis)

densities.df = data.frame(x1=grid.points[,1],x2=grid.points[,2],density = exp(out_dens$log_densities))
print(ggplot() + geom_tile(data = densities.df, aes(x=x1, y=x2, fill=density)) + 
        scale_fill_viridis(discrete=FALSE) + ggtitle("Estimated density"))

#Simulate from the estimated distribution
simulated.data = simulation_b(list_boosting = out, # simply use the output of the boosting function 
                              size = 1000 # size of simulation
)

plot(simulated.data[,1], simulated.data[,2], xlim = c(0,1), ylim = c(0,1),xlab="x1",ylab="x2", main="simulated data")
