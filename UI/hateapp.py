import streamlit as st
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import cv2
import os
import joblib
import pickle
from PIL import Image
import pytesseract
from moviepy.editor import VideoFileClip
import base64
from io import BytesIO

# Load model and vectorizer
model = joblib.load("ensemble_classifier.joblib")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define labels
class_labels = {
    0: 'Hate Speech',
    1: 'Offensive Language',
    2: 'Not Offensive Language'
}

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# Function to classify text
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    tfidf_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(tfidf_text)
    return class_labels.get(prediction[0], 'Unknown')

# Function to convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    sound = AudioSegment.from_file(audio_path, format="wav")
    chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS-14, keep_silence=500)
    text = ""
    for i, chunk in enumerate(chunks):
        chunk.export(f"chunk{i}.wav", format="wav")
        with sr.AudioFile(f"chunk{i}.wav") as source:
            audio_listened = recognizer.record(source)
            try:
                text += recognizer.recognize_google(audio_listened)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
    return text

# Function to convert video to audio and then extract text
def video_to_text(video_path):
    audio_path = 'temp_audio.wav'
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)
    return audio_to_text(audio_path)

# Streamlit UI
st.set_page_config(page_title='HateShield', page_icon='🗣️')

st.markdown("""
    <style>
        .stApp {
            background-color: #182628;
        }
        .stTextArea textarea {
            background-color: #f2f2f2;
            color: #006064;
        }
        .stFileUploader {
            background-color: #fce4ec;
            color: #880e4f;
        }
        .stButton button {
            background-color: #3b945e;
            color: #182628;
        }
        .stButton button:hover {
            background-color: #245a43; /* Button hover background color */
            color: #ffffff; /* Button hover text color */
        }
        .prediction-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 20px;
            margin: 20px 0;
        }
        .not-offensive {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .offensive {
            background-color: #ffcdd2;
            color: #b71c1c;
        }
    </style>
""", unsafe_allow_html=True)


# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load your logo image
logo_image = Image.open("static\logo.png")
logo_base64 = image_to_base64(logo_image)

# Embed the logo in the HTML
st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" alt="RespectRadar Logo" style="width: 160px; height: auto; margin-bottom: 10px; ">
        <h1>HateShield</h1>
        <h4><i>Unmask online intolerance</i></h4>
    </div>
    """, unsafe_allow_html=True)
st.write("")

st.markdown("<p style='text-align: center;'>This hate speech detection application helps make online spaces safer by identifying and mitigating harmful content in text, audio, and video. This application aims to promote a healthier, more respectful online environment.</p>", unsafe_allow_html=True)
st.write("")
st.write("")
st.markdown("""
    <style>
    .button-container {
        display: flex;
        justify-content: right;
    }
    .button-container .stButton {
        margin: 0px 0px 0px 30px; /* Adjust margin as needed */
    }
    </style>
    """, unsafe_allow_html=True)

#button definition and functionalities
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Text Input"):
        st.session_state.input_type = "text"
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Audio Input"):
        st.session_state.input_type = "audio"
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Video Input"):
        st.session_state.input_type = "video"
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("About the Project"):
        st.session_state.input_type = "about"
    st.markdown('</div>', unsafe_allow_html=True)
st.write("")
st.write("")


# Display the relevant input fields and outputs based on the selected input type
if 'input_type' in st.session_state:
    if st.session_state.input_type == "text":
        st.header('Text Input')
        text_input = st.text_area('Enter text to classify:')
        if st.button('Classify Text'):
            if text_input.strip():
                prediction = classify_text(text_input)
                if prediction == 'Not Offensive Language':
                    st.markdown(f'<div class="prediction-box not-offensive">{prediction}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box offensive">{prediction}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter some text to classify.")

    elif st.session_state.input_type == "audio":
        st.header('Audio Input')
        audio_file = st.file_uploader('Upload an audio file', type=['wav', 'mp3'])
        if audio_file is not None:
            audio_path = f'uploaded_audio/{audio_file.name}'
            with open(audio_path, 'wb') as f:
                f.write(audio_file.getbuffer())
            st.write("Processing audio file... (This may take a moment)")
            audio_text = audio_to_text(audio_path)
            st.write(f'Extracted Text from Audio: {audio_text}')
            audio_prediction = classify_text(audio_text)
            if audio_prediction == 'Not Offensive Language':
                st.markdown(f'<div class="prediction-box not-offensive">{audio_prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box offensive">{audio_prediction}</div>', unsafe_allow_html=True)

    elif st.session_state.input_type == "video":
        st.header('Video Input')
        video_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov'])
        if video_file is not None:
            video_path = f'uploaded_videos/{video_file.name}'
            with open(video_path, 'wb') as f:
                f.write(video_file.getbuffer())
            st.write("Processing video file... (This may take a moment)")
            video_text = video_to_text(video_path)
            st.write(f'Extracted Text from Video: {video_text}')
            video_prediction = classify_text(video_text)
            if video_prediction == 'Not Offensive Language':
                st.markdown(f'<div class="prediction-box not-offensive">{video_prediction}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box offensive">{video_prediction}</div>', unsafe_allow_html=True)
    elif st.session_state.input_type == "about":
        st.header('What is hate speech?')
        st.markdown("Hate speech is communication that attacks a person or group on the basis of attributes such as race, religion, ethnic origin, national origin, disability, or gender identity. It can use offensive language, promote violence, or spread negative stereotypes. Hate speech can be online or offline, and is spread across various media.")
        st.header('About')
        st.markdown('This machine learning application incorporates the Hate Speech Dataset from Kaggle, and has the primary functionality of classifying <b>text, audio</b> and <b>video</b> as hate speech, offensive language, and non offensive language. This project is primarily a <b>Natural Language Processing</b> application, aimed at extracting the context from various media to further classify as hate speech. <b>Extended Gradient Boosting</b> model is used for the final predictions and <b>Term Frequency - Inverse Document Frequency</b> vectorizer is used to vectorize and convert the words to numerical values. The libraries used for audio and video conversion are <b>pydub</b> and <b>moviepy</b> respectively. Audio is first transcribed into text, and then predicted. Video is converted to audio and then transcribed to text. This project is developed using <b>Streamlit</b>.', unsafe_allow_html=True)


#disclaimer and footer
st.markdown(" ")
st.markdown("<h2 align=center> #SayNoToHate<h2>", unsafe_allow_html=True)
st.markdown(" ")
st.markdown(
    """
    <div style="background-color: #245a43;
            color: #ffffff; border: 1px solid #ffffff; padding: 10px; border-radius: 20px;">
        <h6 style='text-align: center;'>Disclaimer</h6>
        <p style='text-align: center;'>
            This application is designed to detect and analyze instances of hate speech in text, audio, and video. 
            During its operation, offensive language may be displayed, thus user discretion is advised.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
st.markdown("<p style='text-align: center; color: #ffffff;'>© 2024 HateShield. All rights reserved.</p>", unsafe_allow_html=True)




