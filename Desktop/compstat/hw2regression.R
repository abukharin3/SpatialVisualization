regular = read.csv("forclosure.csv")
model = lm(regular$price ~., data = regular)
summary(model)

model1 = step(model)
summary(model1)
