import cv2
import os
from PIL import Image
import numpy as np
import pickle

# Taking the current dir name
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
# Taking the image dir
image_dir=os.path.join(BASE_DIR,"images")

#importing Face Classifier 
face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

#Here comes the LBPHFace Recognizer for the training purpose
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.save("trainer.yml")

current_id=0
labels_id={}
x_train=[]
y_labels=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("jfif") or file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)

            label=os.path.basename(os.path.dirname(path))
            #print(label,path) # printing each picture paths from the directory
            
            if label in labels_id:
                pass
            else:
                labels_id[label]=current_id
                current_id+=1



            pil_image=Image.open(path).convert("L") # .convert() into grayscale
            image_array=np.array(pil_image,'uint8') # image converted into numpy array 
            faces=face_cascade.detectMultiScale(image_array)

            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id)

print(labels_id)

with open("labels.pickle","wb") as f:
    pickle.dump(labels_id,f)

recognizer.train(x_train,np.array(y_labels))
