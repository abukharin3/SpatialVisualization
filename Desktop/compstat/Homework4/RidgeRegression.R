library(ridge)
library(glmnet)

# Process Data
data = read.csv("Short.csv")
response = data$Price
X = as.matrix(data)



# Range of lambda values
lambdas <- 10^seq(2, -2, by = -.1)

#Fit ridge regression
fit <- glmnet(X, response, alpha = 0, lambda = lambdas)

#Use cross-validation to evaluate different lambdas
cv_fit <- cv.glmnet(X, response, alpha = 0, lambda = lambdas)
plot(cv_fit)

#Get optimal lambda
opt_lambda <- cv_fit$lambda.min
opt_lambda

#Create the model
model = linearRidge(response ~., data = data, lambda = opt_lambda)
summary(model)
