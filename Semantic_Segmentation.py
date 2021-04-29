import cv2
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter





def run_Semantic_Segmentation() :
    st.subheader('Semantic Segmentation')
    st.write('사전 훈련된 Semantic Segmentation모델에는 U-Net, SegNet, PSPNet, DeepLabv3+, E-Net등이 있다.')
    st.markdown('각 모델별 간단한 설명은 사이드바에서 선택하세요.')
    model = st.sidebar.radio('Model',('U-Net', 'SegNet', 'PSPNet', 'DeepLabv3+', 'E-Net','View Result'))
    if model == 'U-Net' :
        st.subheader('U-Net')
        st.write('U-Net의 기본 아이디어는 업 sampling 연산자가 pooling작업을 대체하는 일반 네트워크에 연속 layer를 추가하는 것이다. 이로인해 U-Net의 layer는 출력의 해상도를 높인다.')
        st.image('data/images/SS1.png')
    elif model == 'SegNet' :
        st.subheader('SegNet')
        st.write('SegNet 아키텍처는 인코더 네트워크, 해당 디코더 네트워크 및 최종 분류 픽셀 단위 계층으로 구성된다.')
        st.image('data/images/SS2.png')
        st.subheader('Encoder')
        st.write('VGG-16에서 13개의 Convolution layer를 가져오는 인코더에서 Convolution 및 MaxPooling이 수행된다. 해당 MaxPooling Index는 2x2 MaxPooling을 수행하는 동안 저장된다.')
        st.subheader('Decoder')
        st.write('Up Sampling 및 Convolution은 디코더의 소프트맥스 분류기에서 각 픽셀의 끝에서 수행된다. 해당 인코더 계층의 MaxPooling Index는 Up Sampling 프로세스 중에 호출되고 Up Sampling된다. 그런 다음 K 클래스 소프트 맥스 분류기를 사용하여 각 픽셀을 예측한다.')
    elif model == 'PSPNet' :
        st.subheader('PSPNet') 
        st.write('주어진 입력 이미지에 대해 Convolution신경망을 사용하여 특징 맵이 추출된다. 그런 다음 피라미드 구문 분석 모듈을 사용하여 하위 영역의 다양한 표현을 수집한다. 그 다음에는 Up Sampling 및 연결 Layer를 통해 로컬 및 글로벌 컨텍스트 정보를 모두 포함하는 피쳐의 최종 표현을 형성한다. 마지막으로, 이전 레이어의 출력이 Convolution Layer에 입력되어 픽셀 당 최정 예측을 얻는다.')
        st.image('data/images/SS3.png')
    elif model == 'DeepLabv3+' :
        st.subheader('DeepLabv3+')
        st.write('DeepLab은 시멘틱 분할 최첨단 모델이다. 이 모델이 출시되기 전에는 필터와 풀링작업을 사용하여 다양한 속도로 다중 규모 상황 정보를 인코딩 할 수 있었다. 새로운 네트워크는 공간 정보를 복구하여 더 날카로운 경계로 물체를 캡처 할 수 있다. 이 모델은 두가지 접근 방식을 결합한다.')
        st.image('data/images/SS4.png')
    elif model == 'E-Net' :
        st.subheader('E-Net')
        st.write('이 모델은 정확도를 높이면서 저전력 모바일 장치에서 실행하는 것을 목표로한다. 기존 모델보다 최대 18배 빠르며, FLOP갯수가 75배 적으며, 매개 변수가 79배 적어 정확도가 훨씬 더 높다.')
        st.image('data/images/SS5.png')

    elif model == 'View Result' : 
        st.write('streamlit에서의 모델링은 EC2(free tier)서버의 성능 문제로 어려워서, local에서 진행한 결과를 자료로 첨부하고자 한다.')
        Type = st.sidebar.radio('Type',('Image','Video'))
        if Type == 'Image' :
            st.video('data/videos/ss-result.mp4')
            st.text('Result Image')
            st.image('data/images/ss_result_image.png')
        if Type == 'Video' :
            st.video('data/videos/ss-result-video.mp4')
            st.write('결과파일을 저장하느라 영상 길이가 길어졌습니다. 결과만 보고싶으신분은 아래 동영상을 재생하세요.')
            st.text('Result Video')
            st.video('data/videos/video_seg.mp4')

