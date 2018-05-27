# Data Preprocessing Template
setwd("C:/Users/Deepa Manu/Machine Learning A-Z/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------")
# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                    ave(dataset$Age, FUN = function(x) mean (x,na.rm = TRUE)), 
                    dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean (x,na.rm = TRUE)), 
                     dataset$Salary)
# Encoding the categorical variables
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain', 'Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                          levels = c('No','Yes'),
                          labels = c(0,1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
# seed is set to 123 [or any other number] to keep the same results
set.seed(123)
# In split, you put only the dependent varaible and in SplitRatio, you put the trainin set ration[different from python]
# Split will have a value = TRUE if it is part of the training set and FALSE if it is part of the test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Feature scaling is done so that all numerical variables 
# are put in the same range so that NO variables dominates the other.
# just scale(training_set) and scale(test_set) will give an error
# that x must be numeric. This is because the country and purchased 
# variables are converted to numberic values, using factors. 
# But, they are actually not numeric. So, apply scaling, excluding those columns
#training_set = scale(training_set)
#test_set = scale(test_set)
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])