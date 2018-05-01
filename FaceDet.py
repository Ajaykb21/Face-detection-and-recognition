# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:33:11 2018

@author: Ajay
"""
import cv2

facedetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)

while(True):
    ret,img=cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break
    
cam.release()
cv2.destroyAllWindows()