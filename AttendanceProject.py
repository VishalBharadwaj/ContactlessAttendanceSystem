import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import os.path
from os import path
import shutil


path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now().strftime("%d/%m/%Y,%H:%M")
            f.write(f'\n{name},{now}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            markAttendance(name)
        else:
            name = 'Unknown'
            # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

#faceLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
#cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0],faceLoc[0],faceLoc[0]),(255,0,255),2)

#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0],faceLocTest[0],faceLocTest[0]),(255,0,255),2)

#results = face_recognition.compare_faces([encodeElon], encodeTest)
#faceDis = face_recognition.face_distance([encodeElon], encodeTest)
