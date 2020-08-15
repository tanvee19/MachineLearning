

"""
Code Challenge: Simple Linear Regression
  Name: 
    Food Truck Profit Prediction Tool
  Filename: 
    Foodtruck.py
  Dataset:
    Foodtruck.csv
  Problem Statement:
    Suppose you are the CEO of a restaurant franchise and are considering 
    different cities for opening a new outlet. 
    
    The chain already has food-trucks in various cities and you have data for profits 
    and populations from the cities. 
    
    You would like to use this data to help you select which city to expand to next.
    
    Perform Simple Linear regression to predict the profit based on the 
    population observed and visualize the result.
    
    Based on the above trained results, what will be your estimated profit, 
    
    If you set up your outlet in Jaipur? 
    (Current population in Jaipur is 3.073 million)
        
  Hint: 
    You will implement linear regression to predict the profits for a 
    food chain company.
    Foodtruck.csv contains the dataset for our linear regression problem. 
    The first column is the population of a city and the second column is the 
    profit of a food truck in that city. 
    A negative value for profit indicates a loss.
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

dataset = pd.read_csv('Foodtruck.csv')
dataset.describe()
plt.boxplot(dataset.values)  
dataset.plot(x='Population', y='Profit', style='o')  
plt.title('Profit food truck')  
plt.xlabel('Population of city')  
plt.ylabel('Profit gained')  
plt.show()

features = dataset.iloc[:, :-1].values  
labels = dataset.iloc[:, 1].values 

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(features, labels) 
#print(regressor.intercept_)  
#print (regressor.coef_)
 
x = [5.2524]
x = np.array(x)
x = x.reshape(1,-1)
labels_pred = regressor.predict(x) 
print(labels_pred)

Score = regressor.score(features, labels)
print(Score)


"""
Code Challenge: Simple Linear Regression

  Name: 
    Box Office Collection Prediction Tool
  Filename: 
    Bahubali2vsDangal.py
  Dataset:
    Bahubali2vsDangal.csv
  Problem Statement:
    It contains Data of Day wise collections of the movies Bahubali 2 and Dangal 
    (in crores) for the first 9 days.
    
    Now, you have to write a python code to predict which movie would collect 
    more on the 10th day.
  Hint:
    First Approach - Create two models, one for Bahubali and another for Dangal
    Second Approach - Create one model with two labels
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

dataset1 = pd.read_csv('Bahubali2_vs_Dangal.csv')

'''
dataset1.describe()
plt.boxplot(dataset.values)  

dataset1.plot(x='Bahubali_2_Collections_Per_day', y='Days', style='o')  
plt.title('BoxOffice')  
plt.xlabel('Bahubali')  
plt.ylabel('Profit gained')  
plt.show()

dataset1.plot(x='Dangal_collections_Per_day', y='Days', style='o')  
plt.title('BoxOffice')  
plt.xlabel('Dangal')  
plt.ylabel('Profit gained')  
plt.show()
'''

features = dataset1.iloc[:, :-2].values  
label1 = dataset1.iloc[:, 1].values 
label2 = dataset1.iloc[:,2].values
from sklearn.linear_model import LinearRegression  
regressor1 = LinearRegression()  
regressor1.fit(features, label1) 
#print(regressor1.intercept_)  
#print (regressor1.coef_)
 
regressor2 = LinearRegression()  
regressor2.fit(features, label2) 
#print(regressor2.intercept_)  
#print (regressor2.coef_)


x = [10]
x = np.array(x)
x = x.reshape(1,-1)
labels_pred = regressor1.predict(x) 
print("Bahubali's profit : ",labels_pred)
labels_pred1 = regressor2.predict(x) 
print("Dangal's profit : ",labels_pred1)


plt.scatter(features, label1,color='red')
plt.scatter(features, label2,color='blue')
plt.plot(features, regressor1.predict(features),color='red') 
plt.plot(features, regressor2.predict(features),color='blue') 
plt.scatter(x,labels_pred,color='green')
plt.scatter(x,labels_pred1,color='green') 
plt.title('BoxOffice')  
plt.xlabel('Movies')  
plt.ylabel('Profit gained')  
plt.show()