import cv2
import pandas as pd
import face_recognition
import os
import csv
from csv import reader
from datetime import datetime

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
    with open('Attendance.csv', 'a', newline='') as response:
        writer = csv.writer(response)

        with open('Attendance.csv', 'r') as read_obj:
            one_char = read_obj.read(1)

            if one_char:
                df = pd.read_csv('Attendance.csv')

                if studentName not in list(df['Name']):
                    data = ["A" for _ in range(len(df.iloc[0]))]
                    data[0] = studentName
                    df.loc[len(df['Name']) + 1] = data

                if f'{datetime.now().strftime("%d/%m/%Y")}' not in next(reader(read_obj)):
                    df[f'{datetime.now().strftime("%d/%m/%Y")}'] = [None for _ in range(len(df['Name']))]
            else:
                writer.writerows([
                    ["Name", f'{datetime.now().strftime("%d/%m/%Y")}'],
                    [studentName, present]]
                )
                df = pd.DataFrame({"Name": [studentName], f'{datetime.now().strftime("%d/%m/%Y")}': [present]})

    df.loc[df['Name'] == studentName, [f'{datetime.now().strftime("%d/%m/%Y")}']] = present
    df.to_csv('Attendance.csv', index = False)


for student in studentList:
    curr_student = cv2.imread(f'{path}/{student}')
    images.append(curr_student)
    studentNames.append(os.path.splitext(student)[0])

studentsEncodeList = encode(images)

unknown_faces = face_recognition.load_image_file('testImages/testImg.jpeg')
unknown_faces = cv2.cvtColor(unknown_faces, cv2.COLOR_BGR2RGB)

unknown_loc = face_recognition.face_locations(unknown_faces)
unknown_encode = face_recognition.face_encodings(unknown_faces)

attendance_dict = {}
present_names = {}

for i in range(0, len(studentsEncodeList)):
    present = False
    for j in range(0, len(unknown_loc)):
        face_match = face_recognition.compare_faces([unknown_encode[j]], studentsEncodeList[i], tolerance=0.54)
        if face_match[0]:
            # attendance_dict[studentNames[i]] = "P"
            markAttendance(studentNames[i], "P")
            present_names[i] = unknown_loc[j]
            present = True
            break
    if not present:
        # attendance_dict[studentNames[i]] = "A"
        markAttendance(studentNames[i], "A")

# for i in present_names:
#     cv2.rectangle(unknown_faces, (present_names[i][0], present_names[i][1]), (present_names[i][2], present_names[i][3]), (255, 0, 255), 2)
#     cv2.putText(unknown_faces, studentNames[i], (present_names[i][0]+6, present_names[i][3]-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
# for i in locs:
#     cv2.rectangle(unknown_faces, (i[0], i[1]), (i[2], i[3]), (255, 0, 255), 2)
# face_match = face_recognition.compare_faces([unknown_encode], studentsEncodeList[3], tolerance=0.54)
# faceDis = face_recognition.face_distance([unknown_encode], studentsEncodeList[3])
# print(face_match, faceDis)
# unknown_loc = face_recognition.face_locations(unknown_faces)[2]
# cv2.rectangle(unknown_faces[0], (unknown_loc[3], unknown_loc[0]), (unknown_loc[1], unknown_loc[2]), (255, 0, 255), 2)
# cv2.imshow("Unknown", unknown_faces)
# cv2.waitKey(0)

