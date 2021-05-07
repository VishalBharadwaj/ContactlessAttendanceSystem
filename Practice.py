
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


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


def markAttendance( name ):
    today = datetime.now().strftime("%d-%m-%Y")
    if not path.exista('ImagesAttendance'+today+'.csv'):
        with open('ImagesAttendance'+today+'.csv', 'w') as file:
            file.write(f'{"name"},{"timestamp"}')

        with open('ImagesAttendance'+today+'.csv', 'r+') as f:
        now = datetime.now().strftime("%d/%m/%Y,%H:%M")
        f.write(f'\n{name},{now}')
    else:
        with open('ImagesAttendance'+today+'.csv', 'r+') as f:
            myDataList = f.readlines()
            row_count = sum(1 for row in myDataList)
            exista = []
            for line in myDataList:
                if name not in line:
                exista.append(name)
            if row_count == len(exista):
                now = datetime.now().strftime("%d/%m/%Y,%H:%M")
                f.write(f'\n{name},{now}')
#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            markAttendance(name)



    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
