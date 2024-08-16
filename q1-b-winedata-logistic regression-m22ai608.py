# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 18:55:17 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 08:24:56 2023

@author: Admin
"""

#importing Libraries
import pandas as pd#library to handle dataframe
import numpy as np#library for scientific computing
import matplotlib.pyplot as plt#library for plots
%matplotlib inline
import warnings#library for skip warning
from datetime import date#library for handle DateTime format
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error,confusion_matrix,accuracy_score
#importing dataset
wine=pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\iitj-mtech-notes\\sem2\\ml-1\\assignments\\CSL7020-ml-assign1-m22ai608\\wine.csv")
#first 5rows
wine.head(5)
wine.describe()
#info
wine.info()
#checking null values
wine.isna().sum()
# view dimensions of dataset
wine.shape
# view summary of dataset
wine.info()
#visualization
# pairplot 
sns.pairplot(wine)

#Data analysis
#segregating the categorical and numerical data

# find categorical variables
categorical = [var for var in wine.columns if wine[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)

#find numerical category(vaiables)
numerical = [var for var in wine.columns if wine[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)


#label encoding
wine['quality'].unique()
#converting categorical data into numerical data
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
wine['quality']= label_encoder.fit_transform(wine['quality'])

wine['quality'].unique()

wine
wine.head()

#Duplicate Values
wine.duplicated()
wine.drop_duplicates()
wine.columns

print("There are", wine.duplicated().sum(),'duplicated values in dataset')

# PANDAS: get column number from column name
wine.columns.get_loc("quality")


#Logistic Regression'

class Logistic_Regression():


  # declaring learning rate & number of iterations (Hyperparametes)
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations



  # fit function to train the model with dataset
  def fit(self, X, Y):

    # number of data points in the dataset (number of rows)  -->  m
    # number of input features in the dataset (number of columns)  --> n
    self.m, self.n = X.shape


    #initiating weight & bias value

    self.w = np.zeros(self.n)
    
    self.b = 0

    self.X = X

    self.Y = Y


    # implementing Gradient Descent for Optimization

    for i in range(self.no_of_iterations):     
      self.update_weights()



  def update_weights(self):

    # Y_hat formula (sigmoid function)

    Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) ))    


    # derivaties

    dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))

    db = (1/self.m)*np.sum(Y_hat - self.Y)


    # updating the weights & bias using gradient descent

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db


  # Sigmoid Equation & Decision Boundary

  def predict(self, X):

    Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))     
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred

#Importing the Dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# getting the statistical measures of the data
wine.describe()
wine['quality'].value_counts()

wine.groupby('quality').mean()

# separating the data and labels
features = wine.drop(columns = 'quality', axis=1)
target = wine['quality']

features
target
# Data Standardlization
scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
standardized_data

features = standardized_data
target = wine['quality']
print(features)
print(target)

###Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(features,target, test_size = 0.2, random_state=2)
print(features.shape, X_train.shape, X_test.shape)


#Training the Model
classifier = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#Model Evaluation 
#accuracy
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score( Y_train, X_train_prediction)
training_data_accuracy 
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score( Y_test, X_test_prediction)
test_data_accuracy
#Predicting the test set result  
y_pred= classifier.predict(X_test)  
y_pred

# calculating confusion matrix
conf_mat = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

#precision score and f1 score
from sklearn.metrics import average_precision_score
average_precision_score(Y_test, y_pred)
#F1 score
from sklearn.metrics import f1_score
# Evaluate the performance of the classifier
f1score = f1_score(Y_test, y_pred, average='weighted')
print(f1score)


"""(c) cross validation for logistic regression"""
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
# Split a dataset into k folds for cross validation
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split 
def log_regr_pred(X_test_prediction):
  preds_for_log_regr = []
  probs_for_log_regr = []
  for feats in X_test_for_log_regr: 
    z = np.dot(feats, weights) + biais
    a = 1 / (1 + np.exp(-z))
    probs_for_log_regr.append(a)
    if a > 0.5:
      preds_for_log_regr.append(1)
    elif a <= 0.5:    
      preds_for_log_regr.append(0)
  return preds_for_log_regr           
def evaluate_algorithm_for_logistic_regression(dataset, algorithm, n_folds):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:		
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy.pop()    
		test_set_standard=sc.transform(test_set)      						
		predicted = algorithm(test_set_standard)
		actual = [row[-1] for row in fold]         
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
n_folds = 5
scores = evaluate_algorithm_for_logistic_regression(dataset, y_pred, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
