import cv2
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

def run_Yolo() :
    st.subheader('Yolo')
    st.write('Yolo는 정확도는 조금 낮더라도, 엄청 빠른 속도를 자랑한다.')
    st.write('streamlit에서의 모델링은 EC2(free tier)서버의 성능 문제로 어려워서, local에서 진행한 결과를 자료로 첨부하고자 한다.')
    
    Type = st.sidebar.radio('Type',('Images','Videos'))
    
    if Type == 'Images' :
        st.video('data/videos/Yolo_image.mp4')
        st.text('Result Images')
        st.image('data/images/3.png')
    if Type == 'Videos' :
        st.video('data/videos/Yolo_video.mp4')
        st.text('Result Video')
        st.video('data/videos/video_ret2.mp4')