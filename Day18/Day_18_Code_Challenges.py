

"""
Q1. (Create a program that fulfills the following specification.)
affairs.csv


Import the affairs.csv file.

It was derived from a survey of women in 1974 by Redbook magazine, in which married women were asked 
about their participation in extramarital affairs.

Description of Variables

The dataset contains 6366 observations of 10 variables:(modified and cleaned)

rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)

age: women's age

yrs_married: number of years married

children: number of children

religious: women's rating of how religious she is (1 = not religious, 4 = strongly religious)

educ: level of education (9 = grade school, 12 = high school, 14 = some college, 
16 = college graduate, 17 = some graduate school, 20 = advanced degree)

occupation: women's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar",
 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 
 6 = professional with advanced degree)

occupation_husb: husband's occupation (same coding as above)

affair: outcome 0/1, where 1 means a woman had at least 1 affair.

    Now, perform Classification using logistic regression and check your model accuracy
    using confusion matrix and also through .score() function.

NOTE: Perform OneHotEncoding for occupation and occupation_husb, since they 
should be treated as categorical variables. Careful from dummy variable trap for both!!

    What percentage of total women actually had an affair?

(note that Increases in marriage rating and religiousness correspond to a decrease in the 
likelihood of having an affair.)

    Predict the probability of an affair for a random woman not present in the dataset. 
    She's a 25-year-old teacher who graduated college, has been married for 3 years, has 1 child, 
    rates herself as strongly religious, rates her marriage as fair, and her husband is a farmer.

Optional

    Build an optimum model, observe all the coefficients.


--------------------------
""""

import sklearn as sk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("affairs.csv")

labels = dataset.iloc[:,-1].values 
features = dataset.iloc[:,:-1].values


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [6,7])
featuresone = onehotencoder.fit_transform(features).toarray()
featuresone = featuresone[:, 1:]

"""onehotencoder = OneHotEncoder(categorical_features = [11])
featuresone = onehotencoder.fit_transform(featuresone).toarray()
featuresone = featuresone[:, 1:]
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featuresone = sc.fit_transform(featuresone)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(featuresone, labels)

probability = classifier.predict_proba(featuresone)
labels_pred = classifier.predict(featuresone)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, labels_pred)

s = classifier.score(featuresone,labels)
print(s)
#or
print(float(cm[0][0]+cm[1][1])/sum(sum(cm)))

print("Actual affair : ",cm[1][0]+cm[1][1])
print("Predicted affair : ",cm[0][1]+cm[1][1])

#l = [0,1,0,0,0,0,0,1,0,0,3,25,3,1,10,16]
l = [3,25,3,1,10,16,4,3]


#True case    l=[3,27,13,3,1,14,3,4]
x = np.array(l)
ohe = onehotencoder.transform(x.reshape(1,-1)).toarray()
o = np.delete(ohe,0)
o = sc.transform(o.reshape(1,-1))
l_pred = classifier.predict(o.reshape(1,-1))
if l_pred[0] == 0:
    print("Dont have an affair")
else:
    print("Have an affair")


"""
Q2. (Create a program that fulfills the following specification.)
mushrooms.csv

Import mushrooms.csv file

This dataset includes descriptions of hypothetical samples corresponding to 23 species of 
gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society
 Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, 
 definitely poisonous, or of unknown edibility and not recommended. This latter class was combined
 with the poisonous one.

 

Attribute Information:

classes: edible=e, poisonous=p (outcome)

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

cap-color: brown=n, buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

 

bruises: bruises=t, no=f

 

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

 

gill-attachment: attached=a,descending=d,free=f,notched=n

 

gill-spacing: close=c,crowded=w,distant=d

 

gill-size: broad=b,narrow=n\

 

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,

green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

 

stalk-shape: enlarging=e,tapering=t

 

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

 

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

 

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

 

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

 

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

 

veil-type: partial=p,universal=u

 

veil-color: brown=n,orange=o,white=w,yellow=y

ring-number: none=n,one=o,two=t

 

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

 

spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

 

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

 

habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

    Perform Classification on the given dataset to predict if the mushroom is edible or poisonous
    w.r.t. it’s different attributes.

(you can perform on habitat, population and odor as the predictors)

    Check accuracy of the model.


"""


import sklearn as sk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("mushrooms.csv")

labels = dataset.iloc[:,0:1].values 
features = dataset.iloc[:,[5,21,22]].values


#dataset.isnull().any(axis=0)

from sklearn.preprocessing import LabelEncoder
labelencoder1 = LabelEncoder()
features[:, 0] = labelencoder1.fit_transform(features[:, 0])


from sklearn.preprocessing import LabelEncoder
labelencoder2 = LabelEncoder()
features[:, 1] = labelencoder2.fit_transform(features[:, 1])


from sklearn.preprocessing import LabelEncoder
labelencoder3 = LabelEncoder()
features[:, 2] = labelencoder3.fit_transform(features[:, 2])


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
featuresone = onehotencoder.fit_transform(features).toarray()


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)



from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.25, random_state = 41)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, p = 2) 
classifier.fit(features_train, labels_train)

pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,pred)

print ("Model Score : "+str(round(classifier.score(features_test,labels_test),3)*100)+"%")


"""

*****
Classification Code Challenge
*****

tree_addhealth.csv

Q1. (Create a program that fulfills the following specification.)

For this Code Challenge, The National Longitudinal Study of Adolescent to Adult Health (Add Health)
 data set, an ongoing (longitudinal) survey study that began in the mid-1990s is used. The project 
 website URL is:

http://www.cpc.unc.edu/projects/addhealth/.

This large data set is available online from the University of North Carolina’s Carolina Population 
Center, http://www.cpc.unc.edu/projects/addhealth/data.

 

Import tree_addhealth.csv

 

The attributes are:

 

BIO_SEX: 1 = male 0 = female    

HISPANIC: 1=Yes,0=No    

WHITE : 1=Yes,0=No

BLACK : 1=Yes,0=No          

NAMERICAN: 1=Yes,0=No                      

ASIAN: 1=Yes,0=No                      

ALCEVR1: ever drank alcohol(1=Yes,0=No)   

marever1: ever smoked marijuana(1=Yes,0=No)    

cocever1: ever used cocaine(1=Yes,0=No)                

inhever1: ever used inhalants(1=Yes,0=No)             

cigavail: cigarettes available in home(1=Yes,0=No)

PASSIST: parents or public assistance(1=Yes,0=No)

EXPEL1: ever expelled from school(1=Yes,0=No)

TREG1: Ever smoked regularly(1=Yes,0=No)

Explanatory Variables:

Age

ALCPROBS1:alcohol problems 0-6

DEP1: depression scale

ESTEEM1: self esteem scale       

VIOL1:violent behaviour scale

DEVIANT1: deviant behaviour scale     

SCHCONN1: school connectedness scale       

GPA1: gpa scale  4 points)

FAMCONCT: family connectedness scale       

PARACTV:parent activities scale

PARPRES:parental presence scale

 

    Build a classification tree model evaluating if an adolescent would smoke regularly or not
    based on: gender, age, (race/ethnicity) Hispanic, White, Black, Native American and Asian, 
    alcohol use, alcohol problems, marijuana use, cocaine use, inhalant use, availability of
    cigarettes in the home, depression, and self-esteem.

    Build a classification tree model evaluation if an adolescent gets expelled or not from school
    based on their Gender and violent behavior.
    Use random forest in relation to regular smokers as a target and explanatory variable 
    specifically with Hispanic, White, Black, Native American and Asian.

(Please make confusion matrix and also check accuracy score for each and every section)

"""


import sklearn as sk  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("tree_addhealth.csv")

for i in data:
    data[i] = data[i].fillna(data[i].mode()[0])
    
    
labels = data.iloc[:,-1].values 
features = data.iloc[:,:-1].values

from sklearn.model_selection import train_test_split
ftr,fte,ltr,lte = train_test_split(features,labels,test_size=.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classi = DecisionTreeClassifier(criterion="entropy",random_state=0)
classi.fit(ftr,ltr)
pred = classi.predict(fte)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(pred,lte)

acc_model_part1 = accuracy_score(lte,pred)

print ("Accuracy Score of the Model part 1 : "+str(round(acc_model_part1*100,2))+"%")

fe = data[["BIO_SEX","VIOL1"]].values
la = data["EXPEL1"].values

ftr,fte,ltr,lte = train_test_split(fe,la,test_size=.2,random_state=0)

classi.fit(ftr,ltr)
pred = classi.predict(fte)

cm = confusion_matrix(pred,lte)

acc_model_part2 = accuracy_score(lte,pred)

fe = data[['WHITE','BLACK','HISPANIC','NAMERICAN','ASIAN']].values
la = data["TREG1"].values

ftr,fte,ltr,lte = train_test_split(fe,la,test_size=.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
classi = RandomForestClassifier(n_estimators=10,criterion="entropy", 
                                random_state=0)
classi.fit(ftr,ltr)
pred = classi.predict(fte)


cm = confusion_matrix(pred,lte)

acc_model_part3 = accuracy_score(lte,pred)

print ("Accuracy Score of the Model part 3 : "+str(round(acc_model_part3*100,2))+"%")