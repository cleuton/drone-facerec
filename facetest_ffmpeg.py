import os, sys
import cv2
import easytello.faceprocessor as faceproc

cap = cv2.VideoCapture('udp://0.0.0.0:11111')
imagem = []
while(True):
    if cap.isOpened():
        ret, frame = cap.read()
        image = faceproc.process_frame(frame)
        #print("frame: ",type(frame),frame)
        #print("image: ",type(image),image)
        cv2.imshow('Resultado', image)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap = cv2.VideoCapture('udp://0.0.0.0:11111')
cap.release()
cv2.destroyAllWindows()    