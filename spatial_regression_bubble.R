library(nlme)
library(sp)
library(ridge)

# Procees Data
data = read.csv("COVID_train_data.csv")
test = read.csv("COVID_test_data.csv")

data$X

data[is.na(data)] <- 0
test[is.na(test)] <- 0



Y = data$Death
Lat = data$Latitude
Lng = data$Longitude
cols.dont.want <-c("bedover_mean")
data <- data[, ! names(data) %in% cols.dont.want, drop = F]
cols.dont.want <-c("Death", "X")
cols.dont.want <- c("Death", "X", "bedover_mean", "lag_1", "lag_1",
                    "lag_2", "lag_3", "lag_4", "lag_5", "lag_6", "lag_7", "lag_8", "lag_9", "lag_10",
                    "lag_12", "lag_13", "lag_14", "lag_16", "lag_17", "lag_18", "lag_20",
                    "lag_21", "lag_22", "lag_23", "lag_24", "lag_25", "lag_26", "lag_27", "lag_28", "lag_29",
                    "lag_30", "lag_31", "lag_32", "lag_33", "lag_34", "lag_35", "lag_36", "lag_37", "lag_38", "lag_39",
                    "lag_40", "lag_41", "lag_42", "lag_43", "lag_44", "lag_45", "lag_46", "lag_47", "lag_48",
                    "Mean_Income", "Median_Age", "Population_60.", "ICUbed_mean", "newICU_mean", "icuover_mean")


X <- data[, ! names(data) %in% cols.dont.want, drop = F]
test_X <- test[, ! names(test) %in% cols.dont.want, drop = F]


# Fit linear model
data.lm <- lm(Y~., X)
summary(data.lm)

# Compute Loss
loss = predict(data.lm, test_X) ** 2
avg = sum(loss) / length(loss)
avg

#Fit ridge regression
data.ridge <- linearRidge(Y~., X)
summary(data.ridge)
loss = predict(data.ridge, test_X) ** 2
avg = sum(loss) / length(loss)
avg

# Create covariance matrix and visualize residuals
coordinates(data) <- ~ Latitude + Longitude
data$LinearResiduals <- rstandard(data.lm)
bubble(data, "LinearResiduals")
cor = corExp(form = ~Latitude + Longitude | Date, nugget = FALSE)

# Fit GLS model and fit covariance matrix
data.gls <- gls(Y~., X, correlation = cor, method = "REML")
summary(data.gls)
data$SpatialResid <- resid(data.gls)
bubble(data, "SpatialResid")
loss = predict(data.gls, test_X) ** 2
avg = sum(loss) / length(loss)
avg

df <- data.frame("Resid" = resid(data.gls), "ID" = data$X, "Date" = data$Date)
df
write.csv(df, "residuals1.csv")




