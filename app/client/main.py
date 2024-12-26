import requests
import streamlit as st


SERVER_URL = "http://localhost:8500"
VIDEO_FORMATS = ["asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpeg4", "mpg", "ts", "webm", "wmv"]


st.title("Player Tracking")

video = st.file_uploader("Upload video", VIDEO_FORMATS)

if st.button("Track!"):
    st.video(requests.post(f"{SERVER_URL}/infer", files={"video_file": video}).content)
