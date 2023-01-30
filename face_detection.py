import cv2
#import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
    
from keras import optimizers
from keras.models import load_model

class face_detection:
    def detect_face_from_image(self,img):
        cscpat='C:/python/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
        facecas=cv2.CascadeClassifier(cscpat)

        img=cv2.imread(img)
        #img=load_img(img,target_size=(178, 218))
        print(type(img))
        img=np.array(img)
        #img=np.pad(img,(1,1),'constant')
        
#        cv2.imshow('org',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        print('sssssssssssssssssssssssssssssssss',img.shape)
        
        
        border=cv2.copyMakeBorder(
                img,
                top=10,
                bottom=10,
                left=10,
                right=10,
                borderType=cv2.BORDER_CONSTANT,
                value=[0,0,0])
        
#        cv2.imshow('pad',border)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        g_img=cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('sample',g_img)
        faces=facecas.detectMultiScale(
                g_img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30,30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        if(len(faces)>1 or len(faces)<1):
            return []
            #print(len(faces))
        else:
        
            for (x,y,w,h) in faces:
#                cv2.rectangle(img,(x-40,y-40),(x+w+40,y+h+40),(0,0,255),3)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                
#            cv2.imshow('sample',img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
                
            for (x,y,w,h) in faces:
#                x=x-5
#                y=y-10
#                w=w+10
#                h=h+15
                f_img=img[y:y+h,x:x+w]
                print(type(f_img))
#                cv2.imshow('crop',f_img)
#                
            return f_img
                
#        cv2.imshow('crop',f_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        print('bhuuuuuuuuuuuuunhhunu',f_img.shape)
#        x=img_to_array(img)
#        resized=cv2.resize(x,(218,178))
#        x=resized.reshape((resized.shape[0],resized.shape[1],resized.shape[2]))
#        x=preprocess_input(x)
#        
#        cv2.imwrite('pp1.jpg',x)
#        
#        print('bhuuuuuuuuuuuuunhhunu',resized.shape)
#        cv2.imshow('crop',x)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#
#OBJ=face_detection()
#OBJ.detect_face_from_image('F:\\dataset used\\age\\train\\19-36\\36_1_1_20170109132934818.jpg.chip.jpg')