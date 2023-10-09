# Required packages
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(viridis)
library(boostPM)

#Generate data from a beta mixture
set.seed(11)
n = 10000
d = 2
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
               Omega = cbind(rep(0,d), rep(1, d)), # we can input the information of the sample space, ("Omega" in the paper)
                             # if Omega = NULL, the sample space is automatically set according to the range
               ntree_max_marginal = 100, # # trees per dimension used in the first stage
               ntree_max_dependence = 1000, # # trees used in the second stage
               c0 = 0.1, # c0 = global scale of the learning parameter
               gamma = 0.0, # gamma = stronger regularization for small nodes
               max_resol = 10, # maximum resolution (depth) of trees
               min_obs = 10, # if # obs in a node > min_obs, this node is no longer split
               early_stop = c(1e-5,10), # if it is (1e-5, 50), this means we move to the next step
                                        # when the average improvement given by the recent 50 trees is less than 1e-5
               nbins = 100, # # bins (n_bins-1 = # grid points)
               max_n_var = d, # this is an experimental one so should be set to d 
               # parameters for the PT-based weak learner
               alpha = 0.9, # prior prob of dividing a node = alpha * (1 + depth)^beta
               beta = 0.0,
               precision = 1.0 # precision of the theta prior

)

#Evaluate the log-densities
out_dens = eval_density_b(list_boosting = out, # simply use the output of the boosting function
                          eval_points = grid.points # matrix of evaluation points
                          )


#Visualize the estimated density function
densities.df = data.frame(x1=grid.points[,1],x2=grid.points[,2],density = exp(out_dens$log_densities))
print(ggplot() + geom_tile(data = densities.df, aes(x=x1, y=x2, fill=density)) + 
        scale_fill_viridis(discrete=FALSE) + ggtitle("Estimated density"))

#Simulate from the estimated distribution
simulated.data = simulation_b(list_boosting = out, # simply use the output of the boosting function 
                              size = 1000 # size of simulation
)

plot(simulated.data[,1], simulated.data[,2], xlim = c(0,1), ylim = c(0,1),xlab="x1",ylab="x2", main="simulated data")

