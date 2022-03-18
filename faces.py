import numpy as np
import cv2

#importing Face Classifier 
face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

cap=cv2.VideoCapture(0)

while(True):
    #Capture Frame by Frame
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detecting faces using cascade
    faces=face_cascade.detectMultiScale(gray)

    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        color=(255,0,0)
        thick=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,thick)

    #Display the resulting frame 
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

#when everything done,release the capture 
cap.release()
cv2.destroyAllWindows()

    