import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as f
import torchaudio
import numpy as numpy
from scipy.io.wavfile import read
#load_audio
def load_audio(audiopath, sampling_rate):
    if audiopath[-4:] == '.wav':
        audio, lsr = load_wav_to_torch(audiopath)
    elif audiopath[-4:] == '.mp3':
        audio, lsr = librosa.load(audiopath, sr=sampling_rate)
        audio = torch.FloatTensor(audio)

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '2' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)



##classifier
def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    download_models(['classifier.pth'])
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load('.models/classifier.pth', map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]
st.set_page_config(layout='wide')

def main():

    st.title("Final Year Project")
    #file uploader
    uploaded_file = st.file_uploader("upload an audio file ",type=['mp3'])
    if uploaded_file is not None:
        if st.button("Analyze Audio"):
            col1,col2,col3 = st.columns(2)

            with col1:
                st.info ("your results are below")
                #load andclassify the audio file
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                st.info("Result probability :{result}")
                st.success("The uploaded audio is{result*100:.2f} % likely to be AI Generated.")

            with col2:
                st.info("Your Uploaded audio is below")
                st.audio(uploaded_file)
                #create a waveform
                fig = px.line()
                fig.add_scatter(x=list(range(len(audio_clip.squeeze()))),y=audio_clip.squeeze())
                fig.update_layout(
                    title="Waveform Plot",
                    xaxis_title="Time",
                    yaxis_title="Amplitude"
                )
                st.plotly_chart(fig,use_container_width=True)

            with col3:
                st.info("Disclaimer")
                st.warning("These classifcation or detection mechanism are not always accurate.They should be considered as a strong signal and not the ulimate decision makers.")
                 
if __name__=='__main__':
    main()