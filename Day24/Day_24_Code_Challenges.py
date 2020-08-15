"""
Q1. (Create a program that fulfills the following specification.)

Import Crime.csv File.

    Perform dimension reduction and group the cities using
    k-means based on Rape, Murder and assault predictors.

"""


# PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('crime_data.csv')

features = dataset.iloc[:, [1,2,4]].values
labels = dataset.iloc[:,0].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

features = sc.fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
features = pca.fit_transform(features)

explained_variance = pca.explained_variance_ratio_

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)  
#Now plot it        
plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)

plt.scatter(features[pred_cluster == 0], labels[pred_cluster == 0], c = 'blue', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 1], labels[pred_cluster == 1], c = 'red', label = 'Cluster 2')
plt.scatter(features[pred_cluster == 2], labels[pred_cluster == 2], c = 'green', label = 'Cluster 3')
plt.scatter(features[pred_cluster == 3], labels[pred_cluster == 3], c = 'orange', label = 'Cluster 4')
plt.title('Clusters of datapoints')
plt.xlabel('X Cordindates)
plt.ylabel('Cities')
plt.legend()
plt.show()


"""

Q2. (Create a program that fulfills the following specification.)

The iris data set consists of 50 samples from each of three species of 
Iris flower (Iris setosa, Iris virginica and Iris versicolor).



    Four features were measured from each sample: the length and 
    the width of the sepals and petals, in centimetres (iris.data).
    Import the iris dataset already in sklearn module using the 
    following command:\


from sklearn.datasets import load_iris
iris = load_iris()
iris=iris.data


Reduce dimension from 4-d to 2-d and perform clustering to distinguish 
the 3 species.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
features=iris.data


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
features = pca.fit_transform(features)

explained_variance = pca.explained_variance_ratio_

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)  
#Now plot it        
plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
pred_cluster = kmeans.fit_predict(features)
plt.scatter(features[pred_cluster == 0, 0], features[pred_cluster == 0, 1], c = 'blue', label = 'Cluster 1')
plt.scatter(features[pred_cluster == 1, 0], features[pred_cluster == 1, 1], c = 'red', label = 'Cluster 2')
plt.scatter(features[pred_cluster == 2, 0], features[pred_cluster == 2, 1], c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'yellow', label = 'Centroids')
plt.title('Clusters of datapoints')
plt.xlabel('X Cordindates')
plt.ylabel('Y Cordinates')
plt.legend()
plt.show()


"""
Q3. Code Challenge -
Data: "data.csv"

This data is provided by The Metropolitan Museum of Art Open Access
1. Visualize the various countries from where the artworks are coming.
2. Visualize the top 2 classification for the artworks
3. Visualize the artist interested in the artworks
4. Visualize the top 2 culture for the artworks
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
d = dataset['Country'].value_counts().head(20)

plt.bar( d.index, d.values)
plt.xticks(rotation = 90)
plt.show()

d1 = dataset['Classification'].value_counts().head(2)

plt.bar( d1.index, d1.values)
plt.xticks(rotation = 90)
plt.show()

d2 = dataset['Artist Display Name'].value_counts().head(30)

plt.bar( d2.index, d2.values)
plt.xticks(rotation = 90)
plt.show()

d3 = dataset['Culture'].value_counts().head(2)

plt.bar( d3.index, d3.values)
plt.xticks(rotation = 90)
plt.show()