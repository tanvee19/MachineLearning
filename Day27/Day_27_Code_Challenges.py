"""Q1. 

(Click Here To Download Training data File): 
http://openedx.forsk.in/c4x/Forsk_Labs/ST101/asset/Advertisement_training_data.json

(Click Here To Download Test data File):
http://openedx.forsk.in/c4x/Forsk_Labs/ST101/asset/Advertisement_test_data.json


This is the data for local classified advertisements. It has 9 prominent sections:
jobs, resumes, gigs, personals, housing, community, services, for-sale and discussion
 forums. Each of these sections is divided into subsections called categories.
 For example, the services section has the following categories under it:
beauty, automotive, computer, household, etc.

For a set of sixteen different cities (such as newyork, Mumbai, etc.), we provide 
to you data from four sections

        for-sale
        housing
        community
        services

and we have selected a total of 16 categories from the above sections.

        activities
        appliances
        artists
        automotive
        cell-phones
        childcare
        general
        household-services
        housing
        photography
        real-estate
        shared
        temporary
        therapeutic
        video-games
        wanted-housing

Each category belongs to only 1 section.

About Data:

        city (string) : The city for which this Craigslist post was made.
        section (string) : for-sale/housing/etc.
        heading (string) : The heading of the post.

each of the fields have no more than 1000 characters. The input for the program
 has all the fields but category which you have to predict as the answer.

A total of approximately 20,000 records have been provided to you, proportionally 
represented across these sections, categories and cities. The format of training 
data is the same as input format but with an additional field "category", the 
category in which the post was made.

Task:

    Given the city, section and heading of an advertisement, can you predict the 
    category under which it was posted?
    Also Show top 5 categories which has highest number of posts
"""



# Importing the libraries

import pandas as pd

#reading the data from json to dataframe
#as the file contains some special characters in it so to handle that data.find is used
with open("Advertisement_training_data.json") as f:
    data = f.read()
data = data[data.find("["):]

df_train = pd.read_json(data)

#reading  test file
#encoding of file changed to utf-8
with open("Advertisement_test_data.json",encoding = 'utf-8') as f:
    data1 = f.read()
    
data1 = data1[data1.find("["):]

df_test = pd.read_json(data1)


#importing
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#handling heading column of training data using NLP and replace it in same column
from nltk.stem.porter import PorterStemmer
for i in range(0, len(df_train['heading'])):
    review = re.sub('[^a-zA-Z]', ' ', df_train['heading'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if (word=='not') or (not word in set(stopwords.words('english'))) ]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    df_train['heading'][i] = review
    
#handling heading column of testing data using NLP and replace it in same column
from nltk.stem.porter import PorterStemmer
for i in range(0, len(df_test['heading'])):
    review = re.sub('[^a-zA-Z]', ' ', df_test['heading'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if (word=='not') or (not word in set(stopwords.words('english'))) ]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    df_test['heading'][i] = review
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features_train = cv.fit_transform(df_train['heading']).toarray()
features_test = cv.transform(df_test['heading']).toarray()

features_train = pd.DataFrame(features_train)
features_train['city'] = df_train['city']

features_train['section'] = df_train['section']

features_test = pd.DataFrame(features_test)
features_test['city'] = df_test['city']
features_test['section'] = df_test['section']

labels_train = df_train.iloc[:,0].values
 


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features_train['city'] = labelencoder.fit_transform(features_train['city'])
features_test['city'] = labelencoder.transform(features_test['city'])

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features_train['section'] = labelencoder.fit_transform(features_train['section'])
features_test['section'] = labelencoder.transform(features_test['section'])


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1500,1501])
features_train = onehotencoder.fit_transform(features_train).toarray()

features_test = onehotencoder.transform(features_test).toarray()


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(labels_test, labels_pred)

score = classifier.score(features_test,labels_test)



#part 2

count = df_train['category'].value_counts().head(5)
#count1 = df_test['category'].value_counts().head(5)
import matplotlib.pyplot as plt
plt.pie(count.values,explode = None,labels = count.index)


"""

Q2. Facial Recognition + OpenCV Python

Facial recognition is a biometric software application capable of uniquely 
identifying or verifying a person by comparing and analyzing.

Things that you need in this project: OpenCV and face_recognition

The project is mainly a method for detecting faces in a given image by using 
OpenCV-Python and face_recognition module. The first phase uses camera to capture
 the picture of our faces which generates a feature set in a location of your PC.

• The face_recognition command lets you recognize faces in a photograph or folder
 full for photographs.

It has two simple commands

Face_ recognition- Recognise faces in a photograph or folder full for photographs.
face_detection - Find faces in a photograph or folder full for photographs.
For face recognition, first generate a feature set by taking few image of your face 
and create a directory with the name of person and save their face image.


Then train the data by using the Face_ recognition module.By Face_ recognition module
 the trained data is stored as pickle file (.pickle).

By using the trained pickle data, we can recognize face.

The main flow of face recognition is first to locate the face in the picture and the
 compare the picture with the trained data set.If the there is a match, it gives the
 recognized label.
(Ref: https://github.com/sriram251/-face_recognition)

"""


import imutils
import numpy as np
import pickle
import cv2
import face_recognition


def main(): 
    encoding = "F:\\opencv\\face recognisation\\encodings\\encoding1.pickle"
    data = pickle.loads(open(encoding, "rb").read())
    print(data)
    cap = cv2.VideoCapture(0)
  
    if cap.isOpened :
        ret, frame = cap.read()
    else:
         ret = False
    while(ret):
      ret, frame = cap.read()
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      rgb = imutils.resize(frame, width=400)
      r = frame.shape[1] / float(rgb.shape[1])

      boxes = face_recognition.face_locations(rgb, model= "hog")
      encodings = face_recognition.face_encodings(rgb, boxes)
      names = []
   
      for encoding in encodings:
                matches = face_recognition.compare_faces(np.array(encoding),np.array(data["encodings"]))
                name = "Unknown"
               
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                   
                    
                    for i in matchedIdxs:
                                  name = data["names"][i]
                                  counts[name] = counts.get(name, 0) + 1
                                  name = max(counts, key=counts.get)
                names.append(name)
                
      for ((top, right, bottom, left), name) in zip(boxes, names):
          top = int(top * r)
          right = int(right * r)
          bottom = int(bottom * r) 
          left = int(left * r)
          cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
          y = top - 15 if top - 15 > 15 else top + 15
          cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
      cv2.imshow("Frame", frame)
      if cv2.waitKey(1) == 27:
                  
          break                                                

    cv2.destroyAllWindows()

    cap.release()
if __name__ == "__main__":
    main()



    

import numpy as np
import os
import cv2

face_cascade = cv2.CascadeClassifier('F:\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
path = "C:\Users\Tanvee\Desktop\ForskML\Day27"
# path were u want store the data set
id = input('enter user name')

try:
    # Create target Directory
    os.mkdir(path+str(id))
    print("Directory " , path+str(id),  " Created ") 
except FileExistsError:
    print("Directory " , path+str(id) ,  " already exists")
sampleN=0;

while 1:

    ret, img = cap.read()
    frame = img.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        sampleN=sampleN+1;

        cv2.imwrite(path+str(id)+ "\\" +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.waitKey(100)

    cv2.imshow('img',img)

    cv2.waitKey(1)

    if sampleN > 40:

        break

cap.release()

cv2.destroyAllWindows()
