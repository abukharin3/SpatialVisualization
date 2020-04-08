library(glmnet)

# Process Data
data = read.csv("ShortSale.csv")
response = data$Price
X = as.matrix(data)


# Range of lambda values
lambdas <- 10^seq(2, -2, by = -.01)
lambdas

#Fit lasso regression
fit <- glmnet(X, response, alpha = 1, lambda = lambdas)
fit

#Use cross-validation to evaluate different lambdas
cv_fit <- cv.glmnet(X, response, alpha = 0, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min
opt_lambda

#Fit model with best lambda
fit <- glmnet(X, response, alpha = 1, lambda = 27)
coef(fit)
