from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np 
import torch
# import pyttsx3

# def SpeakText(command):
	
# 	# Initialize the engine
# 	engine = pyttsx3.init()
# 	engine.say(command)
# 	engine.runAndWait()


model = YOLO('best.pt')
#cam = 'http://192.168.0.101:4747/video'
cam = 0
cap = cv2.VideoCapture(cam)
cap.set(3, 640)
cap.set(4, 480)


while True:
    objects = []

    ret, img = cap.read()

    if not ret:
        print("Failed to read camera frame")
        break
    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].to(dtype=torch.float)  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            #print(model.names[int(c)])
            annotator.box_label(b, model.names[int(c)])
            objects.append(model.names[int(c)])
          
    img = annotator.result()
    print(objects)  
    #SpeakText(objects)
    cv2.imshow('YOLO V11 Detection', img)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    # if cv2.waitKey(1) & 0xFF == ord('s'):
    #     SpeakText(objects)

cap.release()
cv2.destroyAllWindows()