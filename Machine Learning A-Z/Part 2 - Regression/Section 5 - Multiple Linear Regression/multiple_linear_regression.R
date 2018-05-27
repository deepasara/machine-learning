# Multiple Linear Regression

setwd("C:/Users/Deepa Manu/Machine Learning A-Z/Part 2 - Regression/Section 5 - Multiple Linear Regression")
# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
# Profit ~ . means that Profit is linearly dependent on all
# independent variables
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Since summary(regressor) showed that Profit has significant dependence
# only on R.D.Spend,
# Try out the model with Profit dependnt only on R.D.Spend
regressor_RD = lm(formula = Profit ~ R.D.Spend,
                  data = training_set)

# Building Optimal model using Backword elimination
# Select  significant level as 0.05, ie. 5%
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# Since summary showed State2 and State3 had highest p-value,
# remove State column and fit the model again
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

# Since summary showed Administrtaion had highest p-value 0.602,
# remove Administration column and fit the model again
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)

# Since summary showed Marketing Spend had highest p-value 0.06,
# remove Administration column and fit the model again
regressor = lm(formula = Profit ~ R.D.Spend ,
               data = dataset)
summary(regressor)
