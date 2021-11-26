import cv2
import pandas as pd
import face_recognition
import os
import csv
from datetime import datetime
import time

path = 'trainImages'
images = []
studentNames = []
studentList = os.listdir(path)


def encode(images):
    encodeList = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(image)[0]
        encodeList.append(encodeimg)

    return encodeList


def markAttendance(studentName, present):
    with open('Attendance.csv', 'a', newline=''):
        file = open('Attendance.csv')
        one_char = file.read(1)

        if one_char:
            df = pd.read_csv('Attendance.csv')

            if studentName not in list(df['Name']):
                data = ["A" for _ in range(len(df.iloc[0]))]
                data[0] = studentName
                df.loc[len(df['Name']) + 1] = data

            if f'{datetime.now().strftime("%d/%m/%Y")}' not in next(csv.reader(file)):
                df[f'{datetime.now().strftime("%d/%m/%Y")}'] = [None for _ in range(len(df['Name']))]
        else:
            df = pd.DataFrame({
                "Name": [studentName], 
                f'{datetime.now().strftime("%d/%m/%Y")}': [present]
            })

        file.close()

    df.loc[df['Name'] == studentName, [f'{datetime.now().strftime("%d/%m/%Y")}']] = present
    df.to_csv('Attendance.csv', index = False)

def captureImage():
    while True:
        url='http://192.168.0.102:8080/shot.jpg'
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("testImages/testMain.png", frame)
        cap.release()
        time.sleep(8)
        break

def main():
    # captureImage()
    for student in studentList:
        curr_student = cv2.imread(f'{path}/{student}')
        images.append(curr_student)
        studentNames.append(os.path.splitext(student)[0])

    studentsEncodeList = encode(images)

    unknown_faces = face_recognition.load_image_file('testImages/testImg7.jpeg')
    unknown_faces = cv2.cvtColor(unknown_faces, cv2.COLOR_BGR2RGB)

    unknown_loc = face_recognition.face_locations(unknown_faces)
    unknown_encode = face_recognition.face_encodings(unknown_faces)

    present_names = {}

    for i in range(0, len(studentsEncodeList)):
        present = False

        for j in range(0, len(unknown_loc)):
            face_match = face_recognition.compare_faces([unknown_encode[j]], studentsEncodeList[i], tolerance=0.48)

            if face_match[0]:
                markAttendance(studentNames[i], "P")
                present_names[i] = unknown_loc[j]
                present = True
                break

        if not present:
            markAttendance(studentNames[i], "A")

main()