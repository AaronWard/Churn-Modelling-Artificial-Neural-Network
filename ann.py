# Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----------------Data Preprocessing--------------------

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
#people from 3rd to 12th collumn
X = dataset.iloc[:, 3:13].values
#Exited people (1 or 0)
y = dataset.iloc[:, 13].values 

# Encoding categorical data to numeric data
# because ANN's can only work with numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#After this step, you will see the countries become numbers
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  ------------------Making the ANN -------------------------
# Importing Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

# Add input layer and hidden layer no.1
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))


# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
