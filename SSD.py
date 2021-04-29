import cv2
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

def run_SSD() :
    st.subheader('SSD')
    st.write('SSD의 알고리즘은 각 피처맵에서 다른 스케일과 비율로 defalut box를 생성하고 모델을 통해 계산된 좌표와 클래스 값을 활용해 bounding box를 생성하는 것이다.')
    st.write('streamlit에서의 모델링은 EC2(free tier)서버의 성능 문제로 어려워서, local에서 진행한 결과를 자료로 첨부하고자 한다.')
    
    Type = st.sidebar.radio('Type',('Images','Videos'))
    
    if Type == 'Images' :
        st.video('data/videos/Images.mp4')
        st.text('Result Images')
        st.image('data/images/1.png')
        st.image('data/images/2.png')
    elif Type == 'Videos' :
        st.video('data/videos/Videos.mp4')
        st.text('Result Video')
        st.video('data/videos/video_ret.mp4')