# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the Country categorcal value
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Encoding the Gender categorcal value
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# To create dummy variables to tell that there is no order for the categorical 
# variables ie. Germany is not higher than France etc.
# Also, to avoid dummy variable trap, we need to consider 1 less variable for encoding
# Since we have 2 dummy variables, consider only 1
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Also, to avoid dummy variable trap, we need to consider 1 less variable for country
# The below code will take only 2 columns for country variable and not 3.
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
# Sequqnetial package is used to initialize the Artificial Neural Network
from keras.models import Sequential
# Dense package is used to add the fully connected network in the Artificial Neural Network
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# input_dim is necessary only for the first hidden layer. It is the same as number of independent variables
# Subsequent hidden layers will know how many input layers are needed
# Rectifier activation function is selected for hidden layers
# Sigmoid activation function is chosen for the output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
#In this example, there is only one dependent variable, so out_dim for output layer is 1
# In case there are multiple dependent variable for the classification problem,
# the output_dim should be equal to the number of dependent categorical variables.
# Also, the activation function should be softmax instead of sigmoid
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimizer parameter finds the optimal weights
# loss = 'binary_cross entropy' for only one categorical variable output
# loss = 'categorical_cross entropy' for only one categorical variable output
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch_size specifies the number of observations after which the weights needs to be updated.
# nb_epoch is the rounds the whole dataset is going to be trained over the ANN model
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)