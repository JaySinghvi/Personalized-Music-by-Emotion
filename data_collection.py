import mediapipe as mp
import numpy as np
import pandas as pd
import cv2

name = input("Enter the name of the data : ")
cap = cv2.VideoCapture(0) #to capture video using webcam

#holistic function takes in the frame and returns all the key facial points and hand gestures
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils


x=[] #this contains list of rows
data_size = 0 #this is to check how big is our data

while True:
    lst = [] #this contains all the landmarks
    _, frm = cap.read() #reading the frame
    frm = cv2.flip(frm , 1) #to remove the mirror effect

    res = holis.process(cv2.cvtColor(frm , cv2.COLOR_BGR2RGB)) # converting the holis frame to rgb as cv2 reads in bgr format

    if res.face_landmarks: #if there is a face in the frame it iterates through the different landmarks
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x) #we are storing it with reference to 1st landmark point [1]
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks: #if left_hand is in the frame it iterates through the different landmarks
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x) #we are storing it with reference to 8th landmark point [8]
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42): #if left_hand not in frame
                lst.append(0.0)

        if res.right_hand_landmarks: #if right_hand is in the frame it iterates through the different landmarks
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        x.append(lst)
        data_size = data_size + 1 #to calculate the data size



    #to draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #to let the use know how much data is collected

    cv2.imshow("window", frm) #displaying the frame

    if cv2.waitKey(1) == 27 or data_size > 99: #if user presses esc key then close the window
        cv2.destroyAllWindows()
        cap.release()
        break

np.save(f'{name}.npy', np.array(x))
print(np.array(x).shape)

