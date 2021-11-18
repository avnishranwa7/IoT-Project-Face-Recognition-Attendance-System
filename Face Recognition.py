import cv2
import numpy as np
import face_recognition

imgDivtri = face_recognition.load_image_file('trainImages/Bhanu Soni.jpg')
imgDivtri = cv2.cvtColor(imgDivtri, cv2.COLOR_BGR2RGB)
imgBhanu = face_recognition.load_image_file('testImages/testImg3.jpeg')
imgBhanu = cv2.cvtColor(imgBhanu, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgDivtri)[0]
encodeDivtri = face_recognition.face_encodings(imgDivtri)[0]
cv2.rectangle(imgDivtri, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgBhanu)[0]
encodeTest = face_recognition.face_encodings(imgBhanu)[0]
cv2.rectangle(imgBhanu, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeDivtri], encodeTest, tolerance=0.54)
faceDis = face_recognition.face_distance([encodeDivtri], encodeTest)
print(results, faceDis)

cv2.imshow('Divyanshu Tripathy', imgDivtri)
cv2.imshow('Bhanu Soni', imgBhanu)
cv2.waitKey(0)