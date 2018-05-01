# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 00:05:17 2018

@author: Ajay
"""

import cv2

facedetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
id= input("Enter the id")
Sample=0
while(True):
    ret,img=cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        Sample+=1
        cv2.imwrite("Dataset/user."+str(id)+"."+str(Sample)+".jpg",gray[y:y+h,x:x+h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(200)
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(Sample> 20):
        break
    
cam.release()
cv2.destroyAllWindows()