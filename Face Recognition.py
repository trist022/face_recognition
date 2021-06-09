import os
from face_recognition.api import face_encodings, face_locations
import numpy as np
import cv2
import face_recognition

#Them path
path ="Dataset"
listpath = os.listdir(path)
print(f'So anh la {len(listpath)} anh')
imglist = []
listname = []

#Them path test
testpath = "ImagesTest"
Test_listpath = os.listdir(testpath)
testlist = []

#read img tu listpath
for i in listpath:
    imgIndex = cv2.imread(f'{path}/{i}')
    imglist.append(imgIndex)
    listname.append(os.path.splitext(i)[0])

#read test img
for i in Test_listpath:
    TestIndex = cv2.imread(f'{testpath}/{i}')
    testlist.append(TestIndex)

#encoding dataset
def Find_Encoding(img):
    encodelist = []
    for i in img:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encodelist.append(encode)
    return encodelist


#Encoding test Images
# def Test_Encoding(img):
#     encodelist_Test = []
#     for i in img:
#         i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(i)[0]
#         encodelist_Test.append(encode)
#     return encodelist_Test

#encode data images
encodeListData = Find_Encoding(imglist)

# #encode test images
# encode_TestList = Test_Encoding(testlist)
for im in testlist:
    ims = cv2.resize(im, (0,0), fx=0.25, fy=0.25)
    ims = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    face_loc = face_recognition.face_locations(ims)
    face_encoding = face_recognition.face_encodings(ims,face_loc)
    for encoding, facelocation in zip(face_encoding,face_loc):
        matches = face_recognition.compare_faces(encodeListData,encoding)
        faceDis = face_recognition.face_distance(encodeListData,encoding)
        print(matches)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        Unknow = "CHUA BIET"
        if matches[matchIndex]:
            name = listname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = facelocation
            y1, x2, y2, x1 = y1*1,x2*1,y2*1,x1*1
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(im,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(im,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else:
            y1,x2,y2,x1 = facelocation
            y1, x2, y2, x1 = y1*1,x2*1,y2*1,x1*1
            cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(im,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(im,Unknow,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow(name,im)
    cv2.waitKey(0)
        