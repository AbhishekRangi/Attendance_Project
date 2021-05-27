import cv2
import numpy as np
import face_recognition
import os
#FACE RECOGNITION + ATTENDANCE PROJECT | OpenCV Python (2020)
from datetime import datetime

path = 'A:/ToComplete/face_detection/test photo/'
images = []
classNames = []

#myList = os.glob(path + '/*.jpg')
myList = os.listdir(path)
print(myList)
for cl in myList:
  curImg = cv2.imread(f'{path}/{cl}')
  images.append(curImg)
  classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
  encodeList=[]
  for img in images:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encode=face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

def markAttendance(name):
  with open('A:/ToComplete/face_detection/Attendance.csv','w+') as f:
    myDataList = f.readlines()
    nameList = []
    for line in myDataList:
      entry = line.split(',')
      nameList.append(entry[0])
    if name not in nameList:
      now = datetime.now()
      dtString = now.strftime('%H:%M:%S')
      f.writelines(f'\n{name},{dtString}')
      print(now)
#markAttendance('Elon')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
  success ,img = cap.read()
  imgS = cv2.resize(img,(0,0),None,0.5,0.5)
  imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
  
  
  facesCurFrame = face_recognition.face_locations(imgS)
  encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

  for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
    print(faceDis)
    matchIndex = np.argmin(faceDis)
    if(min(faceDis)<0.52):
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*2 ,x2*2,y2*2,x1*2

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-36),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+3,y2-3),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

  cv2.imshow("00",img)
  cv2.waitKey(1)


