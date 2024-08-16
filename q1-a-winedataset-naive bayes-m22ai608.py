# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 08:24:56 2023

@author: Admin
"""
########################################q1-a-winedataset-naive bayes-m22ai608###################################################

#importing Libraries
import pandas as pd#library to handle dataframe
import numpy as np#library for scientific computing
import matplotlib.pyplot as plt#library for plots
%matplotlib inline
import warnings#library for skip warning
from datetime import date#library for handle DateTime format
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn import naive_bayes
from sklearn import metrics

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

##########################with Library's###################################

#naive bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Split the dataset into training and testing sets
X = wine.iloc[:, 0:11]
X.head(1)
y = wine.iloc[:, 11]
y.head(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Evaluate the performance of the classifier
y_pred = gnb.predict(X_test)
y_pred
#accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

#confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

#Precision ,Recall and F1 Score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# create plot
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


#precision recall curve
from sklearn.metrics import precision_recall_curve 
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
# create plot
plt.plot(precision, recall, label='Precision-recall curve')
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')
_ = plt.legend(loc="lower left")


from sklearn.metrics import average_precision_score
average_precision_score(y_test, y_pred)
#F1 score
from sklearn.metrics import f1_score
# Evaluate the performance of the classifier
f1score = f1_score(y_test, y_pred, average='weighted')
print(f1score)



 # fit a naive_bayes.MultinomialNB() model to the data
    model = naive_bayes.MultinomialNB()
    model.fit(X_train, y_train)
    print(); print(model)

    # make predictions
    expected_y  = y_test
    predicted_y = model.predict(X_test)
    
    # summarize the fit of the model
    print(); print('naive_bayes.MultinomialNB(): ')
    print(); print(metrics.classification_report(expected_y, predicted_y))
    print(); print(metrics.confusion_matrix(expected_y, predicted_y))    



#################### without Library's####################


#Importing libraries & reading dataset

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error,confusion_matrix,accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
winedata=pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\iitj-mtech-notes\\sem2\\ml-1\\assignments\\CSL7020-ml-assign1-m22ai608\\wine.csv")
winedata


#converting categorical data into numerical data
# Encoding categorical variable
winedata['quality_cat'] = winedata['quality'].astype('category').cat.codes
winedata.head()

plt.figure(figsize=(20,10))
plt.subplots_adjust(left=0, bottom=0.5, right=0.9, top=0.9, wspace=0.5, hspace=0.8)
plt.subplot(141)
plt.title('Percentage of good and bad quality wine',fontsize = 20)
winedata['quality'].value_counts().plot.pie(autopct="%1.1f%%")



#Segregating dependent and independent variables
X = winedata.drop('quality_cat',axis=1)
Y = winedata['quality_cat']

"""(a) & (c) Implement Naive Bayes classifier algorithm from scratch with cross validation technique for two class classification."""

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		i=0  
		for row in csv_reader:
			if not row:
				continue
			if i==0:
				i=i+1         
				continue        
			dataset.append(row)
	return dataset
# Calculate the mean of a list of numbers
def mean(numbers):
	 
	return sum(numbers)/float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance) 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		if value=="good":
			lookup[value]=1	 
		elif value=="bad":
			lookup[value]=0	 
		print('[%s] => %d' % (value, lookup[value]))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities
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
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
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
			row_copy[-1] = None							
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)
# Make a prediction with Naive Bayes on Wine Dataset        
#seed(1)
filename = 'C:\\Users\\Admin\\OneDrive\\Desktop\\iitj-mtech-notes\\sem2\\ml-1\\assignments\\CSL7020-ml-assign1-m22ai608\\wine.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1) 
# evaluate algorithm
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
# fit model
training_set=dataset[0:1279]
test_set=dataset[1279:1599]
model = summarize_by_class(training_set)

"""(d) Accuracy, Confusion matrix, ROC curve and F1 Score for Naive Bayes classifier algorithm"""

y_pred_for_naive_bayes=list()
y_test_for_naive_bayes=list()
for row in test_set:
  y_pred_for_naive_bayes.append(predict(model, row))
  y_test_for_naive_bayes.append(row[11])

#Plotting confusion matrix for Naive Bayes classifier algorithm
mat = confusion_matrix(y_test_for_naive_bayes, y_pred_for_naive_bayes)
plt.figure(figsize=(10, 8))
sns.heatmap(mat,xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'], fmt='.0f',annot=True)

y_prob_for_naive_bayes=[]
# Predict the class for a given row
def predictprob(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return probability
for row in test_set:  
  y_prob_for_naive_bayes.append(predictprob(model, row))
fpr, tpr, _ = metrics.roc_curve(y_test_for_naive_bayes,  y_prob_for_naive_bayes)
#create ROC curve
plt.plot(fpr,tpr,marker='.', label= 'ROC curve for NB')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc='lower right')
plt.show()

#accuracy for naive bayes
Naive_Bayes_Accuracy_Score=metrics.accuracy_score(y_test_for_naive_bayes, y_pred_for_naive_bayes)
Naive_Bayes_Accuracy_Score

#Publishing f1 score for Naive Bayes classifier algorithm
f1ScoreForNB = f1_score(y_test_for_naive_bayes, y_pred_for_naive_bayes)
print('F1 score: %f' % f1ScoreForNB)



#######################q1-b-winedata-logistic regression-m22ai608####################################
