import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

model = load_model("model.h5")
label = np.load("lables.npy")

#holistic function takes in the frame and returns all the key facial points and hand gestures
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) #to capture video using webcam

#print(model.summary())
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

        lst = np.array(lst).reshape(1, -1)
        pred = label[np.argmax(model.predict(lst))] #to find the max index and convert to string

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    #to draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.imshow("window", frm) #displaying the frame

    if cv2.waitKey(1) == 27: #if user presses esc key then close the window
        cv2.destroyAllWindows()
        cap.release()
        break
