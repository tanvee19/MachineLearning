"""
Q1. (Create a program that fulfills the following specification.)
deliveryfleet.csv


Import deliveryfleet.csv file

Here we need Two driver features: mean distance driven per day (Distance_feature) 
and the mean percentage of time a driver was >5 mph over the speed limit 
(speeding_feature).

    Perform K-means clustering to distinguish urban drivers and rural drivers.
    Perform K-means clustering again to further distinguish speeding drivers 
    from those who follow speed limits, in addition to the rural vs. urban division.
    Label accordingly for the 4 groups.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('deliveryfleet.csv')
features = dataset.iloc[:, [1, 2]].values

plt.scatter(features[:,0], features[:,1])
plt.show()

from sklearn.cluster import KMeans
"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)    

#Now plot it        
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)

#plt.scatter(features[:,0][y_kmeans == 0], features[:,1][y_kmeans == 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Rural')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Urban')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('Distance')
plt.ylabel('speed')
plt.legend()
plt.show()


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)

#plt.scatter(features[:,0][y_kmeans == 0], features[:,1][y_kmeans == 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Rural follow speed limit')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Urban follow speed limit')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Urban do not follow speed limit')
plt.scatter(features[pred_cluster == 3, 0], features[pred_cluster == 3, 1], c = 'black', label = 'Rural do not follow speed limit')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('Distance')
plt.ylabel('speed')
plt.legend()
plt.show()



"""


Q1. (Create a program that fulfills the following specification.)
tshirts.csv


T-Shirt Factory:

You own a clothing factory. You know how to make a T-shirt given the height 
and weight of a customer.

You want to standardize the production on three sizes: small, medium, and large.
 How would you figure out the actual size of these 3 types of shirt 
 to better fit your customers?

Import the tshirts.csv file and perform Clustering on it to make sense out of 
the data as stated above.


"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('tshirts.csv')
features = dataset.iloc[:, [1, 2]].values

plt.scatter(features[:,0], features[:,1])
plt.show()

from sklearn.cluster import KMeans
"""
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)    

#Now plot it        
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

"""

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)

#plt.scatter(features[:,0][y_kmeans == 0], features[:,1][y_kmeans == 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Medium')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Large')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Small')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('Distance')
plt.ylabel('speed')
plt.legend()
plt.show()



"""
Code Challenge - 
 This is a pre-crawled dataset, taken as subset of a bigger dataset 
 (more than 4.7 million job listings) that was created by extracting data 
 from Monster.com, a leading job board.
 
 
 
 Remove location from Organization column?
 Remove organization from Location column?
 
 In Location column, instead of city name, zip code is given, deal with it?
 
 Seperate the salary column on hourly and yearly basis and after modification
 salary should not be in range form , handle the ranges with their average
 
 Which organization has highest, lowest, and average salary?
 
 which Sector has how many jobs?
 Which organization has how many jobs
 Which Location has how many jobs?
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# Importing the dataset (Bivariate Data Set with 3 Clusters)
dataset = pd.read_csv('monster_com-job_sample.csv')

dataset1 = dataset[['location','organization']]

dataset1['organization'] = dataset1['organization'].fillna("None") 

def loc_city(value):
    city = re.findall(r'[A-Z]{2}',value)
    if len(city)==1:
        return city[0]
    else:
        return None
dataset1['city'] = dataset1['location'].apply(loc_city)

def Zip(value):
    city = re.findall(r'[0-9]{5}',value)
    if len(city)==1:
        return city[0]
    else:
        return None    
dataset1['zipcode'] = dataset1['location'].apply(Zip)


def Town(value):
    city = re.findall(r'[A-Z]{1}[a-z]+',value)
    if len(city)==1:
        return city[0]
    else:
        return None  
dataset1['town'] = dataset1['location'].apply(Town)
        

"""

# foe checking that org column dont have any loction value
c=0

for i in range(len(dataset1['organization'])):
    #print(dataset1['organization'][i])
    str1 = dataset1['organization'][i].split(',')
    if len(str1)==2:
        if(re.findall(r'^[\W]{1}[A-Z]{2}[\W]{1}[0-9]{5}',str1[1])):
            print(str1)
            c+=1
            
"""

# for swapping values
for i in range(len(dataset1['organization'])):
    str1 = dataset1['organization'][i].split(',')
    if len(str1)==2:
        if(re.findall(r'^[\W]{1}[A-Z]{2}[\W]{1}[0-9]{5}',str1[1])):
            if dataset1['location'][i] == dataset1['organization'][i]:
                dataset1['location'][i],dataset1['organization'][i]  = dataset1['organization'][i],"None"
                
            else:
                dataset1['organization'][i],dataset1['location'][i] = dataset1['location'][i],dataset1['organization'][i]














