from logging import exception
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import streamlit as st
import random
from PIL import Image, ImageEnhance, ImageFilter

def load_image(Image_file) :
    image = Image.open(Image_file)
    return image

def run_open_cv() :

    st.subheader("image_enhance")

    uploaded_files = st.file_uploader("이미지 업로드",type=['png','jpeg','jpg'] )

    if uploaded_files is not None :
        img = Image.open(uploaded_files)
        st.text('Original Image')
        st.image(img)

    enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Resize","Crop",
                                    "Rotation","Affine","Hue Histogram","Contrast","Brightness",
                                    "Blurring","Convolution","Sharpen","Canny"])
    
    
    try :
        if enhance_type == 'Gray-Scale':
            new_img = np.array(img.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            st.text('Gray image')
            st.image(gray)


        elif enhance_type == 'Resize' :
            scaleX = st.sidebar.slider('X축 비율 입력', 0.1, 3.0)
            scaleY = st.sidebar.slider('Y축 비율 입력', 0.1, 3.0)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            img_resize = cv2.resize(img, None, fx=scaleX, fy=scaleY, interpolation=cv2.INTER_LINEAR)
            st.text('Resize image')
            st.image(img_resize)

        elif enhance_type == 'Crop' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            cr_x1 = st.sidebar.slider('x축 시작 부분', 1, img.shape[1])
            cr_x2 = st.sidebar.slider('x축 끝 부분', 1, img.shape[1])
            cr_y1 = st.sidebar.slider('y축 시작 부분', 1, img.shape[0])
            cr_y2 = st.sidebar.slider('y축 끝 부분', 1, img.shape[0])
            crop_img = img[ cr_y1:cr_y2 , cr_x1:cr_x2 ]
            st.text('Crop image')
            st.image(crop_img)

        elif enhance_type == 'Rotation' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            center = (img.shape[1]/2, img.shape[0]/2)
            rotationAngle = st.sidebar.slider('회전할 각도 입력', 0.0, 360.0)
            scaleFactor = 1
            rotationMatrix = cv2.getRotationMatrix2D( center, rotationAngle, scaleFactor)
            rot_img = cv2.warpAffine(img, rotationMatrix, (img.shape[1], img.shape[0]))
            st.text('Rotation image')
            st.image(rot_img)

        elif enhance_type == 'Affine' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            img_tri_1 = st.sidebar.slider('img_tri_1',0.0, 3.5)
            img_tri_2 = st.sidebar.slider('img_tri_2',0.0, 3.5)
            img_tri_3 = st.sidebar.slider('img_tri_3',0.0, 3.5)
            dst_tri_1 = st.sidebar.slider('dst_tri_1',0.0, 3.5)
            dst_tri_2 = st.sidebar.slider('dst_tri_2',0.0, 3.5)
            dst_tri_3 = st.sidebar.slider('dst_tri_3',0.0, 3.5)
            warpMat = np.float32( [img_tri_1, img_tri_2, img_tri_3, dst_tri_1, dst_tri_2, dst_tri_3])
            warpMat = warpMat.reshape(2,3)
            affine_img = cv2.warpAffine(img, warpMat, ( int(img.shape[1]*1.5), int(img.shape[0]*1.5)  )  )
            st.text('Affine transfrom image')
            st.image(affine_img)

        elif enhance_type == 'Hue Histogram' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            st.text('Hue Histogram image')
            st.image(hsv_img)

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

        elif enhance_type == 'Convolution' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            kernel_size = st.sidebar.slider('커널 사이즈 입력', 1, 25)
            kernel = np.ones( ( kernel_size, kernel_size ) ) / kernel_size**2
            conv_img = cv2.filter2D(img, -1, kernel)
            st.text('Convolutuon image')
            st.image(conv_img)

        elif enhance_type == 'Sharpen' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            sharpen = np.array([
                        [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]] , dtype='int' )
            sharpen_image = cv2.filter2D(img, -1, sharpen)
            st.text('Sharpen image')
            st.image(sharpen_image)

        elif enhance_type == 'Canny' :
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2RGBA)
            highThreshold = st.sidebar.slider('highThreshold', 100, 1000)
            lowThreshold = st.sidebar.slider('lowThreshold', 50, 1000)
            apertureSizes = [3, 5, 7]
            apertureIndex = st.sidebar.slider('apertureIndex',0, 2, step=1)
            blurAmount = st.sidebar.slider('blurAmount',0 , 20)
            if blurAmount > 0 :
                blurredSrc = cv2.GaussianBlur(img, (2*blurAmount +1, 2*blurAmount+1), 0)    
            else :
                blurredSrc = img.copy()
            appertureSize = apertureSizes[apertureIndex]
            canny_img = cv2.Canny(blurredSrc, lowThreshold, highThreshold, apertureSize=appertureSize)
            st.text("Canny image")
            st.image(canny_img)

    except Exception:
        st.subheader("이미지를 업로드 하시거나 알맞은 범위를 선택해 주시기 바랍니다.")