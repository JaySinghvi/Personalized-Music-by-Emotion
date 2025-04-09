# 🎧 Emotion-Based Music Recommender

A real-time emotion-driven music recommender that uses facial and hand gesture recognition to detect user mood and suggest personalized songs from YouTube.

## 🚀 Overview

This project blends computer vision with sentiment analysis to recommend music based on real-time user emotions. Built using TensorFlow, Keras, and OpenCV, the system recognizes facial expressions and hand gestures, classifies emotional states, and fetches mood-aligned songs. An interactive Streamlit UI allows users to set language or artist preferences for an engaging, personalized music experience.

## 🧠 Methodology

- **Facial Emotion Recognition**: Used pre-trained CNN models (TensorFlow/Keras) to classify emotions from facial expressions via webcam.
- **Hand Gesture Recognition**: Experimented with hand signs to refine emotion predictions and improve sentiment confidence.
- **Sentiment-Music Mapping**: Mapped detected emotions (e.g., happy, sad, neutral) to corresponding music moods and YouTube query logic.
- **Streamlit UI**: Designed an intuitive web interface where users can input language or artist preferences.
- **Real-Time Integration**: Triggered YouTube search results via the `pywhatkit` library based on detected emotions.

## 📊 Features

- Real-time webcam-based facial and hand recognition  
- Sentiment-to-music mapping with contextual filters  
- Support for language and artist preferences  
- Lightweight, responsive Streamlit UI  
- Uses YouTube as a scalable music backend  

## 🛠️ Tech Stack

- Python  
- TensorFlow & Keras  
- OpenCV  
- Streamlit  
- pywhatkit  
- NumPy & Pandas  
- Matplotlib (for debugging and visualization)

## ▶️ How to Run

### 📁 Clone the Repository
`git clone https://github.com/JaySinghvi/Sentiment-Driven-Music-Recommender.git
cd Sentiment-Driven-Music-Recommender`

### 📦 Install Required Packages
`pip install -r requirements.txt`

### 🎬 Run the App
`streamlit run app.py`
