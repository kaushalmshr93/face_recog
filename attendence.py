import cv2
import numpy as np
import face_recognition
import os
from imutils.video import VideoStream

PATH = "C:/Users/Gs-1551/PycharmProjects/FacialRecog/ImagesBasic"
images = []
classnames = []
myList = os.listdir(PATH)
print(myList)

for cls in myList:
    curImg = cv2.imread(f'{PATH}/{cls}')
    images.append(curImg)
    classnames.append(os.path.splitext(cls)[0])
print(classnames)

def findEncoding(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList

encodeListKnown = findEncoding(images)
print("Encoding DONE")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesInFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, facesInFrame)

    for encodeFace, faceLoc in zip(encodeCurrFrame, facesInFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img,(x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
