library(softImpute)

data= read.csv("movie_r.csv")
x = as.matrix(data)
x = as.matrix(x)
dim(x)
x[x == 0] = NA

try = softImpute(x) #Compute NA values
c = complete(x, try) #substitute for na

