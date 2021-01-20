#Generate the data
set.seed(10)
n = 1000
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

plot(data[,1], data[,2], xlab = "x1", ylab = "x2", xlim=c(0,1), ylim=c(0,1), main = "Observations")

#Make the grid points
x.grid = seq(0.005,0.995,by=0.01)
y.grid = seq(0.005,0.995,by=0.01)
grid.points = as.matrix(expand.grid(x.grid,y.grid))

out = boosting(data = data, #Data matrix
         grid.points = grid.points, #Grid points used to compute the predictive density
         max.resol = 5, #Maximum resolution
         n.particle = 100, #Number of particles
         n.grid.L = 32-1, #Number of grid points for the L's prior
         c = 0.1, #Lerning rate
         n.trees = 50, #The number of trees
         out.pred.scores = FALSE, #If TRUE, output the predictive scores
         out.for.simulation = TRUE #If TRUE, output the results necessary for simulation
)

#Visualize the estimated density function
library(ggplot2)
library(viridis)

densities.df = data.frame(x1=grid.points[,1],x2=grid.points[,2],density = out$densities)
print(ggplot() + geom_tile(data = densities.df, aes(x=x1, y=x2, fill=density)) + scale_fill_viridis(discrete=FALSE) + ggtitle("Estimated density"))


#Simulate from the estimated distribution
simulated.data = simulation(out$result.for.simulation, #The information of the measures obtained in the FS algorithm
            1000 #The size of the simulated data
           )

plot(simulated.data[,1], simulated.data[,2], xlim = c(0,1), ylim = c(0,1),xlab="x1",ylab="x2", main="Simulated data")
