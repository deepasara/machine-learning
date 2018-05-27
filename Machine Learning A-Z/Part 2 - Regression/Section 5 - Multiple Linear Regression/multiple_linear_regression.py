# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# 3rd column[State] in independent variable need to be encoded since it a categorical data
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# We are eliminating one of the dummy variable,in this case 
# eliminating California
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# No need to apply feature scaling for multiple linear regression
# since the library takes care of this
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Most of the model libraries consider b0(constant part), but
# statsmodel library does not take into account the constant b0
# So, in order to take care of this, add a column of 1 at the beginning
# of the array X (making x0 = 1)
# array = np.ones((50,1).astype(int)) means create an array of 50 rows and 1 column and convert it to int
# axis = 1 means add it as a column
# first column indicates X0 = 1
X = np.append(arr = np.ones((50, 1)).astype(int), values =X, axis =1)
# Indicate that we are considering the complete set of predictors
# All rows and columns from dependent variable array X
# X_opt is the optimum X array
# We are assuming the significant level = 0.05 ie. 5%
X_opt = X[:,[0,1,2,3,4,5]]
# Fit new array optimum X to models OLS 
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()

# From summary, we can see that independent variable x2 has the highest p value of .990.
# So, as per backward elimination method, we should remove
# independent variable with highest p-value and fit it to the model again
# So, remove independent variale x2 in column 2

X_opt = X[:,[0,1,3,4,5]]
# Fit new array optimum X to models OLS 
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()

# From summary, x1 has the highest p-value of 0.940, So, remove x1 in next cycle
X_opt = X[:,[0,3,4,5]]
# Fit new array optimum X to models OLS 
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()

# from summary, x2 had highest p-values of 0.602, So, remove x2 and fit the model
X_opt = X[:,[0,3,5]]
# Fit new array optimum X to models OLS 
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()

# from summary, x2 had highest p-values of 0.060, So, remove x2 and fit the model
X_opt = X[:,[0,3]]
# Fit new array optimum X to models OLS 
regressor_OLS = sm.OLS(endog =y, exog = X_opt).fit()
regressor_OLS.summary()

# From summary, we can see x1 which represent R&D spend is the 
# significant predictor of profit

