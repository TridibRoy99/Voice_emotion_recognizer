
import streamlit as st
import sounddevice
from scipy.io.wavfile import write
import os
import pandas as pd
import librosa.display
import glob
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import numpy as np
sounddevice.query_devices()

st.set_page_config(
    page_title="Voice Emotion Recognizer",
    page_icon="🧠",
    layout="wide",
    menu_items={
         'Get Help': 'https://www.linkedin.com/in/tridib-roy-974374145/',
         'Report a bug': "https://www.linkedin.com/in/tridib-roy-974374145/",
         'About': "Portfolio WebApp"
     }
)

st.title("The Machine that finally understands you!!")

st.header("Record your own voice")

with st.sidebar:
    values = st.slider('select how long you want to record',5, 30, 5)
    st.write(f"Record for {values} seconds")

if st.button(f"Click to Record"):
        fs=44100
        second=values
        with st.spinner("Recording....."):
            record_voice=(sounddevice.rec(int(second*fs),samplerate=fs,channels=2))
            sounddevice.wait()
            write("output.wav",fs,record_voice)
        
        data, sampling_rate = librosa.load('output.wav')
        plt.figure(figsize=(15, 5))
        image=librosa.display.waveshow(data, sr=sampling_rate)
        plt.savefig("audio_img.jpg")
        
        st.subheader("This is the visual representation of your voice")
        st.image("audio_img.jpg")

        st.subheader("and the recorder audio too...")
        st.audio(("output.wav"))
        
        with st.spinner("Analyzing your voice....."):
            json_file = open('saved_models/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model_v2.h5")
            Sentiments={ 0:'female_angry', 1:'female_calm', 2:'female_fearful', 3:'female_happy', 4:'female_sad',
                        5:'male_angry', 6:'male_calm', 7:'male_fearful', 8:'male_happy',9:'male_sad'}
            X, sample_rate = librosa.load('output.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
            featurelive = mfccs
            livedf2 = featurelive
            livedf2= pd.DataFrame(data=livedf2)
            livedf2 = livedf2.stack().to_frame().T
            twodim= np.expand_dims(livedf2, axis=2)
            livepreds = loaded_model.predict(twodim, 
                                 batch_size=32, 
                                 verbose=1)
            livepreds1=livepreds.argmax(axis=1)
            liveabc = livepreds1.astype(int).flatten()
            emotion = [emotions for (number,emotions) in Sentiments.items() if liveabc == number]
        st.success("Processing Completed!")
        st.subheader(f'It seems you are a **{emotion[0].split("_")[0]}** who is **{emotion[0].split("_")[1]}**')
