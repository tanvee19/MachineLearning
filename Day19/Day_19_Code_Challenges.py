"""
Q1. (Create a program that fulfills the following specification.)
Auto_mpg.txt

Here is the dataset about cars. The data concerns city-cycle fuel consumption in miles per gallon (MPG).

    Import the dataset Auto_mpg.txt
    Give the column names as "mpg", "cylinders", "displacement","horsepower","weight","acceleration", 
    "model year", "origin", "car name" respectively
    Display the Car Name with highest miles per gallon value
    Build the Decision Tree and Random Forest models and find out which of the two is more accurate 
    in predicting the MPG value
    Find out the MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders,
    having acceleration around 22.2 m/s due to it's 100 horsepower engine giving it a displacement
    of about 215. (Give the prediction from both the models)

"""
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
dataset = pd.read_csv("Auto_mpg.txt",sep = "\s+",header = None)
dataset.columns = ["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name"]

dataset["horsepower"] = dataset["horsepower"].replace("?",np.nan)

dataset["horsepower"] = dataset["horsepower"].convert_objects(convert_numeric = True)

dataset["horsepower"]= dataset["horsepower"].replace(np.nan,dataset["horsepower"].mean())

features = dataset.drop(['mpg','car name'], axis=1)  
labels = dataset['mpg']  


from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30)  


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)

"""
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(features_train, labels_train)  

features_test = sc.transform(features_test)
labels_pred = regressor.predict(features_test)
df=pd.DataFrame({'Actual':labels_test, 'Predicted':labels_pred})  
df 
"""

from sklearn.ensemble import RandomForestRegressor
features_test = sc.transform(features_test)
regressor = RandomForestRegressor(n_estimators=25, random_state=0)  
regressor.fit(features_train, labels_train)  
labels_pred = regressor.predict(features_test)  

l = [6,215,100,2630,22.2,80,3]
l = np.array(l).reshape(1,-1)
l = sc.transform(l)
l_pred = regressor.predict(l)  

s = regressor.score(features_train,labels_train)
print(s) 

s = regressor.score(features_test,labels_test)
print(s) 

scikit-learn needs everything to be numerical for decision trees to work.

So, use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2.

"""
Q1. (Create a program that fulfills the following specification.)
PastHires.csv


Here, we are building a decision tree to check if a person is hired or not based on certain predictors.

Import PastHires.csv File.

    Build and perform Decision tree based on the predictors and see how accurate
    your prediction is for a being hired.

Now use a random forest of 10 decision trees to predict employment of specific candidate profiles:

    Predict employment of a currently employed 10-year veteran, previous employers 4, 
    went to top-tire school, having Bachelor's Degree without Internship.
    Predict employment of an unemployed 10-year veteran, ,previous employers 4, didn't
    went to any top-tire school, having Master's Degree with Internship.

"""


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
dataset = pd.read_csv("Pasthires.csv")

labels = dataset.iloc[:,-1].values 
features = dataset.iloc[:,:-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
features[:, 1] = labelencoder1.fit_transform(features[:, 1])


from sklearn.preprocessing import LabelEncoder
labelencoder3 = LabelEncoder()
features[:, 3] = labelencoder3.fit_transform(features[:, 3])


from sklearn.preprocessing import LabelEncoder
labelencoder4 = LabelEncoder()
features[:, 4] = labelencoder4.fit_transform(features[:, 4])


from sklearn.preprocessing import LabelEncoder
labelencoder5 = LabelEncoder()
features[:, 5] = labelencoder5.fit_transform(features[:, 5])



from sklearn.model_selection import train_test_split  
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30)  


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)  
regressor.fit(features_train, labels_train)  
labels_pred = regressor.predict(features_test)  







