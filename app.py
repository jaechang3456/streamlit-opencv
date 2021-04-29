import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import os
from datetime import datetime
import cv2
import random

from open_cv import run_open_cv
from Semantic_Segmentation import run_Semantic_Segmentation
from SSD import run_SSD
from Yolo import run_Yolo

def main():
    
    st.title('OpenCV and Object Detection')
    # 사이드바 메뉴
    menu = ['Home','OpenCV','Semantic Segmentation', 'SSD', 'Yolo']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice =='Home':
        st.write('이 앱은 스트림릿과 연동하여 내 이미지 또는 동영상을 업로드 하여 OpenCv를 이용해 이미지를 처리하고, Object Detection의 간단한 설명과 사용 사례를 보여주는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')
        model = st.sidebar.radio('Explanation', ('OpenCV','SDC','Semantic Segmentation','Object Detection Summary','Object Detection Model'))

        if model == 'OpenCV' :
            st.write('Semantic Segmentation, SSD 등 직접 실습해 본 모델은 사이드바 메뉴에서 결과를 볼 수 있다.')
            st.subheader('OpenCV')
            st.write('영상 관련 라이브러리로서 사실상 표준의 지위를 가지고 있다. 조금이라도 영상처리가 들어간다면 필수적으로 사용하게 되는 라이브러리. OpenCV 이전에는 MIL 등 상업용 라이브러리를 많이 사용했으나 OpenCV 이후로는 웬만큼 특수한 상황이 아니면 OpenCV만으로도 원하는 영상 처리가 가능하다. 기능이 방대하기 때문에 OpenCV에 있는 것만 다 쓸 줄 알아도 영상처리/머신러닝의 고수 반열에 속하게 된다. 조금 써봤다는 사람은 많지만 다 써봤다는 사람은 별로 없으며, 최신 버전의 라이브러리를 바짝 따라가는 사람은 영상 전공자 중에서도 드물다.')
            st.subheader('컴퓨터비전(Computer Vision) 이란?')
            st.markdown('사람이 보는것처럼, 컴퓨터가 이미지를 해석하는 것!')
            st.subheader('왜 OpenCV를 쓸까?')
            st.text('1. 사용하기가 쉽다.')
            st.text('2. 컴퓨터비전에서 가장 많이 쓰이는 라이브러리이다.')
            st.subheader('OpenCV: Open Source Computer Vision Library')
            st.markdown('https://github.com/opencv/opencv')
            st.text('\n')
            st.text('\n')
            st.text('\n')
            st.text('OpenCV의 이미지 처리방식은 사이드바 메뉴에서 선택하세요.')

        if model == 'SDC' :
            st.subheader('SDC(Self-Driving Cars')
            SDC = cv2.imread('data/images/1_vai9P6E5UHrhPmxzFkwWbQ.jpeg')
            SDC = cv2.cvtColor(SDC, cv2.COLOR_BGR2RGB)
            st.image(SDC) 
            st.write('무인 자동차는 자율주행 자동차(SDC), 또는 로봇 자동차로 알려져 있다. 자율 주행 차의 목적은 운전자 없이 자동으로 운전하는 것이다.')
            SDC2 = cv2.imread('data/images/SDC2.png')
            SDC2 = cv2.cvtColor(SDC2, cv2.COLOR_BGR2RGB)
            st.image(SDC2)
            st.write('Computer Vision은 SDC의 눈으로 간주 될 수 있으며 주변 세계가 어떻게 생겼는지 파악하는데 도움이 된다.') 
            st.write('센서 융합은 RADAR, LIDAR 및 LASER와 같은 다양한 센서의 데이터를 통합하여 주변 환경을 더 깊이 이해하는 방법이다.')
            st.subheader('Benefits of SDCs')
            st.write('Greater safety on roads : 자율 주행의 가장 중요한 결과는 안전하지 않은 운전, 즉 졸음운전, 음주운전 등으로 인한 피해를 줄이는 것이다. 또한, 안전벨트를 안한 사용자, 고속주행차 등에 대한 위험도 줄일 수 있다.')
            st.write('Greater indenpendence for those with mobility problems : 완전 자동화는 일반적으로 우리에게 많은 개인적인 자유를 제공한다. 도움이 필요한 사람들, 즉 거동이 불편한 사람들은 더 자립 할 것이다.')
            st.write('Reduced congestion : SDC를 사용하면 교통 정체의 여러 원인을 해결 할 수 있다. 사고가 적다는 것은 고속도로에서 더 적은 백업을 의미한다. 차량사이의 보다 효율적이고 안전한 거리와 정지등은 도로의 전반적인 혼잡을 감소시킬 것이다.')
            st.write('Reduced environmental impact : 대부분의 자율 주행차는 완전 전기식으로 설계 되었기 때문에 자율 주행 차는 연료 소비와 탄소 배출령을 줄여 불필요한 엔진 공회전으로 인한 온실 가스 배출량을 줄이고 연료 소비량을 줄일 수 있다.')
            st.subheader('Deep learning and computer vision for SDCs')
            st.write('자율주행차에 딥러닝과 컴퓨터비전을 이용 할 수 있는 종류에는 Lane detection, Road sign recognition, Pedestrian detection, Traffic light detection, Car detection, Face recognition, Obstacle detection, Cehicle action recognition등이 있다.')


        if model == 'Semantic Segmentation' :
            st.subheader('Semantic Segmentation')
            SS = cv2.imread('data/images/SS.png')
            SS = cv2.cvtColor(SS, cv2.COLOR_BGR2RGB)
            st.image(SS) 
            st.write('Semantic Segmentation은 주어진 class에 속하는 각 픽셀에 레이블을 지정한다. 일반적으로 이러한 클래스는 도로, 표지판, 차, 자전거, 사람 등 이다. Semantic Segmentation의 가장 큰 장점은 픽셀 수준에서 약간의 차이가 식별에 영향을 미치지 않는다는 것이다.')

        if model == 'Object Detection Summary' :
            ob1 = cv2.imread('data/images/detection.png')
            ob1 = cv2.cvtColor(ob1, cv2.COLOR_BGR2RGB)
            ob2 = cv2.imread('data/images/object-dectection-4.jpg')
            ob2 = cv2.cvtColor(ob2, cv2.COLOR_BGR2RGB)
            st.subheader('Object Detection이란?')
            st.write('이미지나 동영상에서 사람, 동물, 차량 등 의미 있는 객체(object)의 종류와 그 위치(bounding box)를 정확하게 찾기 위한 컴퓨터 비전(computer vision) 기술')
            st.write('영상에서 관심 대상을 인식하기 위해 일반적으로 검출 대상에 대한 후보 영역을 찾고 그 후보 영역에 대한 객체의 종류와 위치를 학습된 모델을 통해 예측한다. 이 과정을 위해서 영상 및 영상 내의 객체 종류(class)와 객체 위치(bounding box) 정보가 필요하다. 얼굴, 도로상의 보행자 및 차량 등의 인식에 딥 러닝(deep learning) 기반의 객체 탐지 기술이 많이 이용된다.')
            st.markdown('Object Detection의 결과는 다음과 같이 나온다.')
            st.image(ob1)
            st.image(ob2)
        
        if model == "Object Detection Model" :
            st.subheader('Object Detection Model')  
            st.write('Object Detection에는 대표적으로 One-Stage Detector와 Two Stage-Detector가 있다.')
            st.write('결과적으로 One-Stage Detector비교적 빠르지만 정확도가 낮고, Two Stage-Detector는 비교적 느리지만 정확도가 높다.')
            Detector = st.radio('Detector',('One-Stage Detector', 'Two-Stage Detector'))
            if Detector == 'One-Stage Detector' :
                st.subheader('One-Stage Detector')
                OSD = cv2.imread('data/images/OD-ch1img05.png')
                st.image(OSD)                
                st.write('One-stage Detector는 Classification, Regional Proposal을 동시에 수행하여 결과를 얻는 방법입니다. 그림과 같이 이미지를 모델에 입력 후, Conv Layer를 사용하여 이미지 특징을 추출한다.')

                st.subheader('R-CNN')
                R_CNN = cv2.imread('data/images/OD-ch1img08.png')
                R_CNN = cv2.cvtColor(R_CNN, cv2.COLOR_BGR2RGB)
                st.image(R_CNN)
                st.write('R-CNN은 Selective Search를 이용해 이미지에 대한 후보영역을 생성한다. 생성된 각 후보영역을 고정된 크기로 wrapping하여 CNN의 input으로 사용한다. CNN에서 나온 Feature map으로 SVM을 통해 분류, Regressor을 통해 Bounding-box를 조정한다. 강제로 크기를 맞추기 위한 wrapping으로 이미지의 변형이나 손실이 일어나고 후보영역만큼 CNN을 돌려야하하기 때문에 큰 저장공간을 요구하고 느리다는 단점이 있다.')

                st.subheader('Fast R-CNN')
                Fast_R_CNN = cv2.imread('data/images/OD-ch1img09.png')
                Fast_R_CNN = cv2.cvtColor(Fast_R_CNN, cv2.COLOR_BGR2RGB)
                st.image(Fast_R_CNN)
                st.write('각 후보영역에 CNN을 적용하는 R-CNN과 달리 이미지 전체에 CNN을 적용하여 생성된 Feature map에서 후보영역을 생성한다. 생성된 후보영역은 RoI Pooling을 통해 고정 사이즈의 Feature vector로 추출한다. Feature vector에 FC layer를 거쳐 Softmax를 통해 분류, Regressor를 통해 Bounding-box를 조정한다.')

                st.subheader('Faster R-CNN')
                Faster_R_CNN_1 = cv2.imread('data/images/OD-ch1img10.png')
                Faster_R_CNN_2 = cv2.imread('data/images/OD-ch1img10-2.png')
                Faster_R_CNN_1 = cv2.cvtColor(Faster_R_CNN_1, cv2.COLOR_BGR2RGB)
                Faster_R_CNN_2 = cv2.cvtColor(Faster_R_CNN_2, cv2.COLOR_BGR2RGB)
                st.image(Faster_R_CNN_1)
                st.image(Faster_R_CNN_2)
                st.write('Selective Search 부분을 딥러닝으로 바꾼 Region Proposal Network(RPN)을 사용한다. RPN은 Feature map에서 CNN 연산시 sliding-window가 찍은 지점마다 Anchor-box로 후보영역을 예측한다. Anchor-box란 미리 지정해놓은 여러 개의 비율과 크기의 Bounding-box이다. RPN에서 얻은 후보영역을 IoU순으로 정렬하여 Non-Maximum Suppression(NMS) 알고리즘을 통해 최종 후보영역을 선택한다. 선택된 후보영역의 크기를 맞추기 위해 RoI Pooling을 거치고 이후 Fast R-CNN과 동일하게 진행한다.')

            if Detector == 'Two-Stage Detector' :
                st.subheader('Two-Stage Detector')
                TSD = cv2.imread('data/images/OD-ch1img06.png')
                st.image(TSD)
                st.write('Two-stage Detector는 Classification, Regional Proposal을 순차적으로 수행하여 결과를 얻는 방법이다. 그림과 같이 Region Proposal과 Classification을 순차적으로 실행하는 것을 알 수 있다.')

                st.subheader('YOLO')
                YOLO = cv2.imread('data/images/OD-ch1img11.png')
                YOLO = cv2.cvtColor(YOLO, cv2.COLOR_BGR2RGB)
                st.image(YOLO)
                st.write('Bouning-box와 Class probability를 하나의 문제로 간주하여 객체의 종류와 위치를 한번에 예측한다. 이미지를 일정 크기의 그리드로 나눠 각 그리드에 대한 Bounding-box를 예측한다. Bounding-box의 confidence score와 그리드셀의 class score의 값으로 학습하게 된다. 간단한 처리과정으로 속도가 매우 빠르지만 작은 객체에 대해서는 상대적으로 정확도가 낮다.')

                st.subheader('SSD')
                SSD = cv2.imread('data/images/OD-ch1img12.png')
                SSD = cv2.cvtColor(SSD, cv2.COLOR_BGR2RGB)
                st.image(SSD)
                st.write('각 Covolutional Layer 이후에 나오는 Feature map마다 Bounding-box의 Class 점수와 Offset(위치좌표)를 구하고, NMS 알고리즘을 통해 최종 Bounding-box를 결정한다. 이는 각 Feature map마다 스케일이 다르기 때문에 작은 물체와 큰 물체를 모두 탐지할 수 있다는 장점이 있다.')

                st.subheader('RetinaNet')
                RN = cv2.imread('data/images/OD-ch1img13.png')
                RN = cv2.cvtColor(RN, cv2.COLOR_BGR2RGB)
                st.image(RN)
                st.write('RetinaNet은 모델 학습시 계산하는 손실 함수(loss function)에 변화를 주어 기존 One-Stage Detector들이 지닌 낮은 성능을 개선했다. One-Stage Detector는 많게는 십만개 까지의 후보군 제시를 통해 학습을 진행한다. 그 중 실제 객체인 것은 일반적으로 10개 이내 이고, 다수의 후보군이 background 클래스로 잡힌다. 상대적으로 분류하기 쉬운 background 후보군들에 대한 loss값을 줄여줌으로써 분류하기 어려운 실제 객체들의 loss 비중을 높이고, 그에 따라 실제 객체들에 대한 학습에 집중하게 한다. RetinaNet은 속도 빠르면서 Two-Stage Detector와 유사한 성능을 보입니다.')

    elif choice == 'OpenCV' :
        run_open_cv()    

    elif choice == 'Semantic Segmentation' :
        run_Semantic_Segmentation()

    elif choice == 'SSD' :
        run_SSD()

    elif choice == 'Yolo' :
        run_Yolo()

if __name__ == '__main__':
    main()