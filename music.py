import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
import keras
from keras.models import load_model
import webbrowser

model = load_model("model.h5")
label = np.load("lables.npy")

if label is None or len(label) == 0:
    raise ValueError("Labels not loaded properly. Check lables.npy file.")

#holistic function takes in the frame and returns all the key facial points and hand gestures
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Streamlit session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if not st.session_state["emotion"]:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def __init__(self):
        self.model = model

    def recv(self, frame):
        try:
            frm = frame.to_ndarray(format = "bgr24")

            frm = cv2.flip(frm , 1) #to remove the mirror effect

            res = holis.process(cv2.cvtColor(frm , cv2.COLOR_BGR2RGB)) # converting the holis frame to rgb as cv2 reads in bgr format

            lst = []

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
                try:
                    prediction = model.predict(lst)
                    pred_idx = int(np.argmax(prediction))
                    pred = str(label[pred_idx])
                except Exception as e:
                    print("Prediction failed:", e)
                    pred = "Unknown"
                print(pred)
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                st.session_state["emotion"] = pred

            #to draw landmarks on the frame
            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

            return av.VideoFrame.from_ndarray(frm, format = "bgr24")
        
        except Exception as e:
            print("Error:", e)
            return av.VideoFrame.from_ndarray(frm, format = "bgr24")

st.title("Sentiment-Driven Music Recommender")
lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key = "emotion-key", mode = WebRtcMode.SENDRECV, desired_playing_state = True, async_processing = True, video_processor_factory= EmotionProcessor)

btn = st.button("Recommend me songs") #recommending songs from youtube using emotion recognition
#when button is pressed, streamlit refreshes all the script so we need to save the pred in a file locally and import it later

if btn:
    if not st.session_state["emotion"]:
        st.warning("Please let me capture your emotion to recommend songs")
        st.session_state["run"] = "true"
    else:
        search_query = webbrowser.open(f'https://www.youtube.com/results?search_query={lang}+{st.session_state['emotion']}+songs+{singer}')
        st.markdown(f"[Click here to view your recommendations]({search_query})", unsafe_allow_html=True)
        st.session_state["emotion"] = ""
        st.session_state["run"] = "false"
