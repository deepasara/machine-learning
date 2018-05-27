# Apriori
setwd("C:/Users/Deepa Manu/Machine Learning A-Z/Part 5 - Association Rule Learning/Section 28 - Apriori")

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
# Min support = 4*7/7500 (if we consider items purchased at least 4 times a week)
# set min confidence as at 20% confidence in this example
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])