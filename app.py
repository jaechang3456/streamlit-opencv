import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
from datetime import datetime
import cv2

from open_cv import run_open_cv
from etc import run_etc

def main():
    
    st.title('OpenCV')
    # 사이드바 메뉴
    menu = ['Home','OpenCV','etc']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice =='Home':
        st.write('이 앱은 스트림릿과 연동하여 내 이미지 또는 동영상을 업로드 하여 OpenCv를 이용해 이미지를 처리하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')

    elif choice == 'OpenCV' :
        run_open_cv()    

    elif choice == 'etc' :
        run_etc()

if __name__ == '__main__':
    main()