# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
# Create a matrix of independent variables. in this example, all columns except the last column
X = dataset.iloc[:, :-1].values
# Create a matrix of dependent variables. in this example, only the last column
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Convert categorical variable country to number format instead of country names
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# Create dummy encoders for categorical orders so that the system doesn't consider them as ordered[will not consider Spain > France etc]
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encode Purchased column
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set with test size as 20% of dataset
# random_state is set to zero [or any other number] to keep the same results
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Feature scaling is done so that all numerical variables 
# are put in the same range so that NO variables dominates the other.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# note that only train set is fitted and transformed and 
# no need to apply fit and transform to test set since it is already 
# fitted to the training set
# The below code fits the values to a range of -1 and +1
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# No need to apply feature scaling to y in this case, since the dependent variable values are not ranging from huge numercal value range
