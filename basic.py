import cv2
import numpy as np
import face_recognition

imgKaushal = face_recognition.load_image_file('C:/Users/Gs-1551/PycharmProjects/FacialRecog/ImagesBasic/kaushaltrain.jpg')#[:, :, ::-1]
imgKaushal = cv2.cvtColor(imgKaushal,cv2.COLOR_BGR2RGB)

imgKaushaltest = face_recognition.load_image_file('C:/Users/Gs-1551/PycharmProjects/FacialRecog/ImagesBasic/kaushaltest.jpg')#[:, :, ::-1]
imgKaushaltest = cv2.cvtColor(imgKaushaltest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgKaushal)[0]
encodeKK = face_recognition.face_encodings(imgKaushal)[0]
cv2.rectangle(imgKaushal, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoc2 = face_recognition.face_locations(imgKaushaltest)[0]
encodeKK2 = face_recognition.face_encodings(imgKaushaltest)[0]
cv2.rectangle(imgKaushaltest, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeKK],encodeKK2)
faceDis = face_recognition.face_distance([encodeKK],encodeKK2)
print(results,faceDis)
cv2.putText(imgKaushaltest,f'{results} {round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('kaushal kashyap', imgKaushal)
cv2.imshow('kaushal test', imgKaushaltest)

cv2.waitKey(0)

