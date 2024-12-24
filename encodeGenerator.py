import cv2
import face_recognition
import pickle
import os

folderPath = 'images'
pathList = os.listdir(folderPath)
imgList = []

stuIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    stuIds.append(os.path.splitext(path)[0])


def findEncoding(imgList):
    encodedList = []

    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)

    return encodedList    

print("Encoding has started, gimme a min <3")
encodelistKnown = findEncoding(imgList)

encodeListKnownWithIDs = [encodelistKnown, stuIds]

print("The encoding is DONE!!")

file = open("encodedFile.p", 'wb')

pickle.dump(encodeListKnownWithIDs, file)
file.close()

print("file saved lessgo")
