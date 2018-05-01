# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 02:04:06 2018

@author: Ajay
"""

import os
import cv2
import numpy as np
from PIL import Image

recognizer= cv2.face.LBPHFaceRecognizer_create()
path="Dataset"

def getImagesWithID(path):
     imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
     faces=[]
     ids=[]
     for imagePath in imagePaths:
         faceImg=Image.open(imagePath).convert('L')
         facesNp=np.array(faceImg,"uint8")
         id=int(os.path.split(imagePath)[-1].split(".")[1])
         
         faces.append(facesNp)
         ids.append(id)
         cv2.imshow("Training ",facesNp)
         cv2.waitKey(5)
     return np.array(ids),faces

ids,faces=getImagesWithID(path)
recognizer.train(faces,ids)
recognizer.write("Recognizer/trainingdata.yml")
cv2.destroyAllWindows()