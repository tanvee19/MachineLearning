#!/usr/bin/env python
# coding: utf-8

# In[2]:


def haar_face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(args.weights)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
    return img

def hog_face_detector(img):
    import dlib
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = dlib.get_frontal_face_detector()
    faces_hog = face_cascade(img, 1)  
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        # draw box over face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    return img

print("1. HAAR CASCADE FACE DETECTION\n2. HISTORGRAM ORIENTED FACE DETECTION \n3. EXIT\n>> Choose.....")
ch = int(input())
if(ch==1):
    while 1:
        ret, img = cap.read()
        cv2.imshow('img',haar_face_detector(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
elif(ch==2):
    while 1:
        ret, img = cap.read()
        cv2.imshow('img',hog_face_detector(img))    # too slow
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Program end")


# In[ ]:




