# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:33:11 2018

@author: Ajay
"""
import cv2
import numpy as np
from gtts import gTTS
import os

facedetect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("Recognizer/trainingdata.yml")
id=0
id1=0

font = cv2.FONT_HERSHEY_SIMPLEX

#font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)
while(True):
    ret,img=cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Emma Watson"
        elif(id==3):
            id="Selena Gomez"
        #elif(id==4):
            #id="Maise Williams"
        elif(id==7):
            id="Vishal"
        else:
            id="Ajay"
        cv2.putText(img,str(id),(x,y+h),font,1.5,(0,0,255),2)
    cv2.imshow("Face",img)
    
    if((id!=id1) and False):
        myobj = gTTS(text= "Hello "+id, lang="en", slow=False)
        myobj.save("welcome.mp3")
        os.startfile("welcome.mp3")
    id1=id
    if(cv2.waitKey(1)==ord('q')):
        break
    
cam.release()
cv2.destroyAllWindows()
