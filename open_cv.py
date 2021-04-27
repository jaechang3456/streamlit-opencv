import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

def load_image(Image_file) :
    img = Image.open(Image_file)
    return img

def run_open_cv() :

    st.subheader("image_enhance")

    uploaded_files = st.file_uploader("이미지 업로드",type=['png','jpeg','jpg'] )

    if uploaded_files is not None :
        img = Image.open(uploaded_files)
        st.text('Original Image')
        st.image(img)

    enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
    
    if enhance_type == 'Gray-Scale':
        new_img = np.array(img.convert('RGB'))
        img = cv2.cvtColor(new_img,1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        st.text('gray image')
        st.image(gray)

    elif enhance_type == 'Contrast':
        c_rate = st.sidebar.slider("Contrast",0.5,3.5)
        enhancer = ImageEnhance.Contrast(img)
        img_output = enhancer.enhance(c_rate)
        st.text('Contrast image')
        st.image(img_output)

    elif enhance_type == 'Brightness':
        c_rate = st.sidebar.slider("Brightness",0.5,3.5)
        enhancer = ImageEnhance.Brightness(img)
        img_output = enhancer.enhance(c_rate)
        st.text('Brightness image')
        st.image(img_output)

    elif enhance_type == 'Blurring':
        new_img = np.array(img.convert('RGB'))
        blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
        img = cv2.cvtColor(new_img,1)
        blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
        st.text('Blurring image')
        st.image(blur_img)
        
    elif enhance_type == 'Original':
        st.image(img,width=300)
    else:
        st.image(img,width=300)










