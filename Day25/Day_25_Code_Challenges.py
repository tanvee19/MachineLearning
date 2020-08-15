"""Q1. Human Activity Recognition

Human Activity Recognition with Smartphones

(Recordings of 30 study participants performing activities of daily living)

(Click Here To Download Dataset): 
    https://github.com/K-Vaid/Python-Codes/blob/master/Human_activity_recog.zip



In an experiment with a group of 30 volunteers within an age bracket of 19 to 48 years,
 each person performed six activities (WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS,
 SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist.
 The experiments have been video-recorded to label the data manually.

The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers
 was selected for generating the training data and 30% the test data.

 

Attribute information 

For each record in the dataset the following is provided:

        Triaxial acceleration from the accelerometer (total acceleration) and the estimated 
        body acceleration. 
        Triaxial Angular velocity from the gyroscope.
        A 561-feature vector with time and frequency domain variables.
        Its activity labels.
        An identifier of the subject who carried out the experiment.

Train a tree classifier to predict the labels from the test data set using the following
 approaches:

  (a) a decision tree approach,

  (b) a random forest approach and

  (c) a logistic regression.

  (d) KNN approach

Examine the result by reporting the accuracy rates of all approach on both the testing 
and training data set. Compare the results. Which approach would you recommend and why?

        Perform feature selection and repeat the previous step. Does your accuracy improve?
        Plot two graph showing accuracy bar score of all the approaches taken with and without
        feature selection.
        
 """
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
 
dataset1 = pd.read_csv('test.csv')
dataset = pd.read_csv('train.csv')


features_train = dataset.drop(['Activity','subject'],axis = 1)
labels_train = dataset['Activity']
features_test = dataset1.drop(['Activity','subject'],axis = 1)
labels_test = dataset1['Activity']

"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
features_train = pca.fit_transform(features_train)
features_test = pca.transform(features_test)
explained_variance = pca.explained_variance_ratio_
"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train, labels_train)


labels_pred = classifier.predict(features_test)
"""
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
"""
s1 = classifier.score(features_test,labels_test)

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test) 

"""
from sklearn.metrics import confusion_matrix  
print(confusion_matrix(labels_test, labels_pred)) 
"""
s2 = classifier.score(features_test,labels_test)
#train the model
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=25, random_state=0)  
classifier.fit(features_train, labels_train)  
labels_pred = classifier.predict(features_test) 

s3 = classifier.score(features_test,labels_test)

# Fitting Logistic Regression to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) #When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
classifier.fit(features_train, labels_train)

#Calculate Class Probabilities
probability = classifier.predict_proba(features_test)

# Predicting the class labels
labels_pred = classifier.predict(features_test)
"""
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
"""
s4 = classifier.score(features_test,labels_test)



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
 
dataset1 = pd.read_csv('test.csv')
dataset = pd.read_csv('train.csv')


features_train = dataset.drop(['Activity','subject'],axis = 1)
labels_train = dataset['Activity']
features_test = dataset1.drop(['Activity','subject'],axis = 1)
labels_test = dataset1['Activity']

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labels_train = labelencoder.fit_transform(labels_train)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
labels_train = onehotencoder.fit_transform(labels_train).toarray()


import statsmodels.api as sm


l = list()
for i in range(len(features_train.columns)):
    l.append(i)
features = sm.add_constant(features_train)   

features_opt = features_train.iloc[:, l].values
classifier_OLS = sm.OLS(endog = labels_train, exog = features_opt).fit()
    #regressor_OLS.summary()
c = list(classifier_OLS.pvalues)

for i in range(len(c)):
    if (c[i] >0.0500000000000000):
        number = np.where(c==c[i])[0]
        l.remove(number)
    else:
        pass




 """       
Q2. Code Challenge

#Online Marketing

(Click Here To Download Resource File) :
    http://openedx.forsk.in/c4x/Manipal_University/FL007/asset/online_marketing.sql

Objective of this case study is to explore Online Lead Conversion for a Life Insurance 
company. Some people are interested in buying insurance products from this company hence
 they visit the site of this Life Insurance Company and fill out a survey asking about 
 attributes like income, age etc. These people are then followed and some of them become
 customers from leads. Company have all the past data of who became customers from lead. 
 Idea is to learn something from this data and when

some new lead comes, assign a propensity of him/her converting to a customer based on 
attributes asked in the survey. This sort of problem is called as Predictive Modelling

Concept:

Predictive modelling is being used by companies and individuals all over the world to 
extract value from historical data. These are the mathematical algorithms, which are 
used to "learn" the patterns hidden on all this data. The term supervised learning or 
classification is also used which means you have past cases tagged or classified 
(Converted to Customer or Not) and you want to use this learning on new data.
 (machine learning)

Here are the attributes of the survey:

Attribute

age (Age of the Lead)

Job (Job Category e.g. Management)

marital (Marital Status)

education (Education of Lead)

smoker (Is Lead smoker or not (Binary – Yes / No))

monthlyincome (Monthly Income)

houseowner (Is home owner or not (Binary – Yes / No))

loan (Is having loan or not (Binary – Yes / No))

contact (Contact type e.g. Cellphone)

mod (Days elapsed since survey was filled)

monthlyhouseholdincome (Monthly Income of all family member)

target_buy (altogether Is converted to customer or not (Binary –Yes /No). This is known
 as Target or Responseand this is what we are modelling.)



Activities you need to perform:



a. Handle the missing data and perform necessary data pre-processing.
b. Summarise the data.
c. Perform feature selection and train using prediction model.
d. For a new lead, predict if it will convert to a successful lead or not.
e. Use different classification techniques and compare accuracy score and
 also plot them in a bar graph.
 
 """
 
  
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

import os
import sqlite3
from pandas import DataFrame
 
with open('online_marketing.sql') as f:
    s = f.read()

l = s.split(';')
conn = sqlite3.connect ( 'online.db' )

c = conn.cursor()
"""
for i in l:
    c.execute(i)
"""
database = pd.read_sql("SELECT * FROM online_marketing",conn)    

database['monthlyincome'] = database['monthlyincome'].replace('',0)

for i in range(len(database)):
    if database['job'][i] == '':
        database['job'][i] = "Not known"
    if database['monthlyincome'][i] == 0:
        database['monthlyincome'][i]  = int(np.mean(database['monthlyincome']))
        
database.describe()



features = database.drop(['target_buy'],axis = 1)
labels = database.iloc[:,-1:-2].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labels_train = labelencoder.fit_transform(labels_train)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
labels_train = onehotencoder.fit_transform(labels_train).toarray()


import statsmodels.api as sm


l = list()
for i in range(len(features_train.columns)):
    l.append(i)
features = sm.add_constant(features_train)   

features_opt = features_train.iloc[:, l].values
classifier_OLS = sm.OLS(endog = labels_train, exog = features_opt).fit()
    #regressor_OLS.summary()
c = list(classifier_OLS.pvalues)

for i in range(len(c)):
    if (c[i] >0.0500000000000000):
        number = np.where(c==c[i])[0]
        l.remove(number)
    else:
        pass