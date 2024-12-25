import cv2
import os
import pickle
import numpy as np
import face_recognition
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)

imgBackground = cv2.imread('resources\\background\\background.png')

# importing the modes images into a list
folderModePath = 'resources/modes'
modePathList = os.listdir(folderModePath)
modesImgList = []

for path in modePathList:
    modesImgList.append(cv2.imread(os.path.join(folderModePath, path)))

#load the encoded file

print("Loading encoded file ....")

file = open('encodedFile.p', 'rb')
encodeListKnownWithIDs = pickle.load(file)
file.close()

encodelistKnown, stuIds = encodeListKnownWithIDs
# print(stuIds)
print("Encoded file loaded !")

modeType = 0
counter = 0
id = -1

while True:
    success, img = cap.read()

    imgS =cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44 +633, 808:808+414] = modesImgList[0]

    for encodedFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches =face_recognition.compare_faces(encodelistKnown, encodedFace)

        faceDist = face_recognition.face_distance(encodelistKnown, encodedFace)

        print("matches: ", matches)
        print("face distance: ", faceDist)

        matchedIndex = np.argmin(faceDist)

        if matches[matchedIndex]:
            # print("known face detected <3 <3 <3")
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            imgBackground= cvzone.cornerRect(imgBackground, bbox, rt=0)
            id = stuIds[matchedIndex]

            if counter == 0:
                counter = 1

    if counter != 0:


        counter += 1            


    cv2.imshow("WEBCAM", img)
    cv2.imshow("Facial Attendance", imgBackground)
    cv2.waitKey(1)

