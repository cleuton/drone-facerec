import os, sys
import cv2
import faceprocessor as faceproc

cap = cv2.VideoCapture(0)
imagem = []
while(cap.isOpened()):
    ret, frame = cap.read()
    image = faceproc.process_frame(frame)
    #print("frame: ",type(frame),frame)
    #print("image: ",type(image),image)
    cv2.imshow('Resultado', image)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    