"""
Q1. Code Challegene (NLP)
Dataset: amazon_cells_labelled.txt


The Data has sentences from Amazon Reviews

Each line in Data Set is tagged positive or negative

Create a Machine learning model using Natural Language Processing that
 can predict wheter a given review about the product is positive or negative

"""


# Importing the libraries
import pandas as pd

# Importing the dataset
# Ignore double qoutes, use 3 
dataset = pd.read_csv('amazon_cells_labelled.txt', delimiter = '\t',header = None)


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer 


corpus = []


for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if (word=='not') or (not word in set(stopwords.words('english'))) ]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
labels = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)

# Fitting Kernel SVM to the Training set
# kernels: linear, poly and rbf
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

# Model Score
score = classifier.score(features_test,labels_test)



"""


Q2. Code Challenge (Connecting Hearts)


Downlaod Link: http://openedx.forsk.in/c4x/Manipal_University/FL007/asset/Resource.zip

What influences love at first sight? (Or, at least, love in the first four minutes?) 
This dataset was compiled by Columbia Business School Professors Ray Fisman and Sheena
 Iyengar for their paper Gender Differences in Mate Selection: Evidence from a Speed
 Dating Experiment.

Data was gathered from participants in experimental speed dating events from 2002-2004.
 During the events, the attendees would have a four minute "first date" with every other
 participant of the opposite sex. At the end of their four minutes, participants were 
 asked if they would like to see their date again.

They were also asked to rate their date on six attributes: Attractiveness, Sincerity,
 Intelligence, Fun, Ambition, and Shared Interests.

The dataset also includes questionnaire data gathered from participants at different
 points in the process.

These fields include: demographics, dating habits, self-perception across key attributes,
 beliefs on what others find valuable in a mate, and lifestyle information.

See the Key document attached for details of every column and for the survey details.


What does a person look for in a partner? (both male and female)


For example: being funny is more important for women than man in selecting a partner! Being 
sincere on the other hand is more important to men than women.

    What does a person think that their partner would look for in them? Do you think
    what a man thinks a woman wants from them matches to what women really wants in 
    them or vice versa. TIP: If it doesn’t then it will be one sided :)

    Plot Graphs for:
            How often do they go out (not necessarily on dates)?
            In which activities are they interested?
    
    If the partner is from the same race are they more keen to go for a date?
    What are the least desirable attributes in a male partner? Does this differ for
    female partners?
    How important do people think attractiveness is in potential mate selection vs. 
    its real impact?
    
    
"""



import pandas as pd


# Importing the dataset
# Ignore double qoutes, use 3 
dataset = pd.read_csv('Dating_Data.csv',delimiter = ',',encoding='Windows 1252')

"""
#part1
d = ['gender','attr1_1','sinc1_1','intel1_1','fun1_1','amb1_1','shar1_1']
dataset0 = dataset[d]

dataset0= dataset0.fillna(0)


dataset1 = dataset0[dataset0['gender']==0]
dataset2 = dataset0[dataset0['gender']==1]

l=list()
for j in dataset1.columns:
    l.append(sum(dataset1[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)


l=list()
for j in dataset2.columns:
    l.append(sum(dataset2[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)

"""
"""
#part2
d = ['gender','attr4_1','sinc4_1','intel4_1','fun4_1','amb4_1','shar4_1']
dataset0 = dataset[d]

dataset0= dataset0.fillna(0)


dataset1 = dataset0[dataset0['gender']==0]
dataset2 = dataset0[dataset0['gender']==1]

l=list()
for j in dataset1.columns:
    l.append(sum(dataset1[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)


l=list()
for j in dataset2.columns:
    l.append(sum(dataset2[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)
"""
#part3
d = ['gender','attr2_1','sinc2_1','intel2_1','fun2_1','amb2_1','shar2_1']
dataset0 = dataset[d]

dataset0= dataset0.fillna(0)


dataset1 = dataset0[dataset0['gender']==0]
dataset2 = dataset0[dataset0['gender']==1]

l=list()
for j in dataset1.columns:
    l.append(sum(dataset1[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)


l=list()
for j in dataset2.columns:
    l.append(sum(dataset2[j]))

l = l[1:]   
import matplotlib.pyplot as plt
plt.bar(d[1:],l)


#vis. Go_Out
f = dataset['go_out'].value_counts()
import matplotlib.pyplot as plt
plt.pie(f.values,explode = None,labels = f.index)


#vis. activities

c = dataset.iloc[:,50:67]

c= c.fillna(0)

l=list()
for j in c.columns:
    l.append(sum(c[j]))
    
import matplotlib.pyplot as plt
plt.pie(l,explode = None,labels = c.columns)
plt.bar(c.columns,l)

#vis samerace
s = dataset.iloc[:,14].value_counts()
import matplotlib.pyplot as plt
plt.pie(s.values,explode = None,labels = s.index)



