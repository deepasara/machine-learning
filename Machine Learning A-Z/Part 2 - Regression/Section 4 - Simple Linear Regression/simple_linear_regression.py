# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# Prepare the independent variable by removi the last column as it is the dependent variable
X = dataset.iloc[:, :-1].values
# Take the last column as it is the dependent variable
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling is not required for Simple Linear Regression
# since the library takes care of scaling by itself
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
# Plot the points of training set as a scatter plot
plt.scatter(X_train, y_train, color = 'red')
# Plot the regression line using predicted values of training set 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# This is the end of the graph. Now, the graph is ready to be plotted
plt.show() # Red points are the actual salaries of the employees. Blue line shows the predicted salaries of the employees by the model

# Visualising the Test set results
# Plot the points of test set as a scatter plot to see how accurate the model is 
plt.scatter(X_test, y_test, color = 'red')
# Note that we are using the same regression line trained by the same training set 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()