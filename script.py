import streamlit as st
from ultralytics import YOLO
import numpy as np 
import cv2

# Page formatting
st.set_page_config(layout="wide")
st.markdown("""
            <style>
                .stApp * {
                    font-size: 1.25rem;
                }
                .stButton>button {
                    height: 2em;
                    width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True)

# Prompting the user to upload a file      
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    left, right = st.columns(2, gap="medium")
    with left:
        image = st.image(uploaded_file)
    
    # Use YOLO
    cv2_image = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8)
                       , cv2.IMREAD_COLOR)
                       
    if (st.button("Analyze")):
        results = YOLO("yolov8n.pt").predict(source=cv2_image)        
        results = results[0]
        
        with left:
            image = image.image(results.plot(), channels="BGR")
            
        # Display list of recognised objects
        with right:
            with st.container(height=300):
                for box in results.boxes.cls:
                    st.write(results.names[box.item()])