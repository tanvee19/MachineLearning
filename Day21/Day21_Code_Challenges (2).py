"""
Q2. This famous classification dataset first time used in Fisher’s classic 1936 paper, 
The Use of Multiple Measurements in Taxonomic Problems. Iris dataset is having 4 features
 of iris flower and one target class.

The 4 features are

SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
The target class

The flower species type is the target class and it having 3 types

Setosa
Versicolor
Virginica
The idea of implementing svm classifier in Python is to use the iris features to train an svm
 classifier and use the trained svm model to predict the Iris species type. To begin with
 let’s try to load the Iris dataset.
 
 """
 
 
import pandas as pd
import numpy as np
from sklearn import datasets

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(x,y, test_size = 0.3, random_state = 0)

 
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)
#predict
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

# Model Score
score = classifier.score(features_test,labels_test)

 
 
 
 
 
 
 
 
 
 