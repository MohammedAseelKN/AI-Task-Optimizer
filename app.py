import streamlit as st
import speech_recognition as sr
import librosa
import numpy as np
import cv2
from deepface import DeepFace
from transformers import pipeline
from task_recommendation import recommend_task

# Load text emotion analysis model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

st.title("AI-Powered Task Optimizer")

# Function to analyze text-based emotion
def analyze_text_emotion(text):
    result = emotion_classifier(text)[0]
    return result['label'], result['score']

# Function to capture an image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        st.error("Could not open webcam")
        return None

    ret, frame = cap.read()
    cap.release()  

    if not ret:
        st.error("Failed to capture image")
        return None

    return frame

# Function to analyze facial expression
def analyze_facial_expression(image):
    cv2.imwrite("captured_face.jpg", image)
    result = DeepFace.analyze("captured_face.jpg", actions=['emotion'])[0]
    return result['dominant_emotion']

# Function to record audio and detect speech-based emotion
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Recording... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    with open("recorded_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    return "recorded_audio.wav"

from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa

# Load the correct pre-trained model
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Function to analyze emotion from audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa

# Load the correct pre-trained model
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Function to analyze emotion from audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa

# Load the correct pre-trained model
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Function to analyze emotion from audio
def analyze_audio_emotion(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    
    # Predict emotion
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits).item()

    # Emotion labels
    emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
    
    detected_emotion = emotion_labels[predicted_class]
    return detected_emotion


# UI Components
st.header("Emotion Detection")

# Text Input (Fixed: Added Unique Key)
text_input = st.text_area("Enter your thoughts:", key="text_input")
if st.button("Analyze Text Emotion"):
    emotion, confidence = analyze_text_emotion(text_input)
    st.write(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
    st.write(f"Recommended Task: {recommend_task(emotion)}")

# Webcam Image Capture
if st.button("Capture Image from Webcam"):
    image = capture_image()
    if image is not None:
        st.image(image, channels="BGR")
        detected_emotion = analyze_facial_expression(image)
        st.write(f"Detected Emotion: {detected_emotion}")
        st.write(f"Recommended Task: {recommend_task(detected_emotion)}")

# Voice Emotion Analysis
if st.button("Record Audio"):
    audio_file = record_audio()
    detected_emotion = analyze_audio_emotion(audio_file)
    st.write(f"Detected Emotion: {detected_emotion}")
    st.write(f"Recommended Task: {recommend_task(detected_emotion)}")
