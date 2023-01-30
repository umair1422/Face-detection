from mtcnn import MTCNN
import cv2




class face_detection:
    def detect_face_from_image(self,img):
        detector = MTCNN('C:/fyp/mtcnn/data/mtcnn_weights.npy')
        image=cv2.imread(img)
        #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ss=image
        result=detector.detect_faces(image)
       
        print(type(result))
        if (len(result)>1 or len(result)==0):
            print('choose single face image')
            return [],[]
        else:
#            cv2.imshow('org',image)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            bounding_box = result[0]['box']
            
            cv2.rectangle(ss,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,0,255),
                          3)
            
#            cv2.imshow("image",ss)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
            print(type(bounding_box))
            for i in bounding_box:
                if(i<0):
                    bounding_box[bounding_box.index(i)]=0
            
            f_img=ss[bounding_box[1]:bounding_box[1]+bounding_box[3]-15 ,bounding_box[0]:bounding_box[0]+bounding_box[2]]
            print(bounding_box[0])
            print(bounding_box[1])
            print(bounding_box[2])
            print(bounding_box[3])
            print(ss.shape)
            
#            cv2.imshow("crop",f_img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            
            return ss,f_img