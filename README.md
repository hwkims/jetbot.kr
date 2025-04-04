![스크린샷(61)](https://github.com/user-attachments/assets/9be17fcd-fb68-430a-b1ba-ba180e38a040)# JetBot: NVIDIA Jetson Nano 기반 오픈소스 AI 로봇

https://docs.google.com/presentation/d/1BvbKuTApiZ13GjLVXmczfvaMWzqVicsF/edit?slide=id.p7#slide=id.p7

## JetBot Project Summaries

JetBot을 활용한 두 가지 AI 프로젝트 관련 자료입니다.

### 1. JetBot AI Road Following (도로 따라가기)

*   **Video Demo (YouTube Short):**
    [![JetBot AI Road Following](https://img.youtube.com/vi/4ZKaz29LizA/0.jpg)](https://www.youtube.com/watch?v=4ZKaz29LizA)
    *(Short Link: https://www.youtube.com/shorts/4ZKaz29LizA)*

*   **Dataset (도로 주행 데이터):** 📊
    *   [https://huggingface.co/datasets/kimhyunwoo/jetbot_road_nvidia/tree/main](https://huggingface.co/datasets/kimhyunwoo/jetbot_road_nvidia/tree/main)

*   **Model (도로 주행 모델):** 🤖
    *   [https://huggingface.co/kimhyunwoo/jetbot_road_following_nvidia/tree/main](https://huggingface.co/kimhyunwoo/jetbot_road_following_nvidia/tree/main)

### 2. JetBot Cup Ramen Following (컵라면 따라가기)

*   **Video Demo (YouTube Short):**
    [![JetBot Cup Ramen Following](https://img.youtube.com/vi/7KwpsvUudrI/0.jpg)](https://www.youtube.com/watch?v=7KwpsvUudrI)
    *(Short Link: https://www.youtube.com/shorts/7KwpsvUudrI)*

*   **Dataset (컵라면 인식 데이터 - Free/Blocked):** 📊
    *   [https://huggingface.co/datasets/kimhyunwoo/jetbot_nvidia_cup_free](https://huggingface.co/datasets/kimhyunwoo/jetbot_nvidia_cup_free)

*   **Model (컵라면 따라가기 모델):** 🤖
    *   [https://huggingface.co/kimhyunwoo/jetbot_nvidia_cup_free/tree/main](https://huggingface.co/kimhyunwoo/jetbot_nvidia_cup_free/tree/main)
    *   *동작 방식:* 컵라면을 찾으면 직진, 못 찾으면 회전하며 탐색합니다.

### 3. JetBot Collision Avoidance (충돌 장애물 회피)

*   **Video Demo (YouTube Short):**
    [![JetBot Collision Avoidance](https://img.youtube.com/vi/fSZEXdUfQNo/0.jpg)](https://www.youtube.com/watch?v=fSZEXdUfQNo)
    *(Short Link: https://www.youtube.com/shorts/fSZEXdUfQNo)*

*   **Dataset (장애물 데이터 - Free/Blocked):** 📊
    *   [https://huggingface.co/datasets/kimhyunwoo/nvidia_jetbot_cv_free_block](https://huggingface.co/datasets/kimhyunwoo/nvidia_jetbot_cv_free_block)

*   **Model (장애물 회피 모델):** 🤖
    *   [https://huggingface.co/kimhyunwoo/nvidia_jetbot_cv_model](https://huggingface.co/kimhyunwoo/nvidia_jetbot_cv_model)
    *   *동작 방식:* 장애물이 앞에 있으면 회전, 못 찾으면 직진하며 탐색합니다.






 



[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub Discussions](https://img.shields.io/github/discussions/NVIDIA-AI-IOT/jetbot)](https://github.com/NVIDIA-AI-IOT/jetbot/discussions)
[![Jetson Forum](https://img.shields.io/badge/Forum-Jetson-green)](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)

 https://hwkims.github.io/jetbot.kr/3d.html
 

JetBot은 NVIDIA Jetson Nano를 기반으로 하는 저렴하고 교육적이며 재미있는 오픈소스 AI 로봇 프로젝트입니다.

## JetBot을 선택해야 하는 이유

*   **합리적인 가격:** Jetson Nano를 포함한 부품 비용이 $250 미만입니다. DIY 키트, 3D 프린팅, 다양한 타사 키트를 지원합니다.
*   **교육적 가치:** 기본 로봇 공학 개념부터 고급 AI 학습까지 다루는 Jupyter Notebook 예제를 제공합니다.
*   **쉬운 설정:** 웹 브라우저에서 프로그래밍하며, Wi-Fi 연결 및 Docker 컨테이너를 통해 간편하게 환경을 구축할 수 있습니다.
*   **재미:** 휴대 가능하고 배터리로 구동되는 AI 컴퓨터와 카메라를 통해 AI, 이미지 처리, 로봇 공학을 탐험할 수 있습니다.

## 시작하기

1.  **차량 선택:** [Bill of Materials (Orin)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/bill_of_materials_orin.md) 또는 [Third Party Kits](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/third_party_kits.md)를 참고하여 하드웨어를 준비합니다.
2.  **JetBot 설정:**
    *   [DIY JetBot Kit](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/getting_started.md#option-1---diy-jetbot-kit)
    *   [Third Party JetBot Kit](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/getting_started.md#option-2---third-party-jetbot-kit)
3.  **소프트웨어 설정:** [Software Setup](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/software_setup.md) (SD 카드 이미지, Docker 컨테이너, 또는 수동 설치)
4.  **예제 따라하기:**
    *   [기본 동작 (Basic Motion)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/examples.md#basic-motion)
    *   [원격 조작 (Teleoperation)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/examples.md#teleoperation)
    *   [충돌 회피 (Collision Avoidance)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/examples.md#collision-avoidance)
    *   [경로 추종 (Road Following)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/examples.md#road-following)
    *   [객체 추종 (Object Following)](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/examples.md#object-following)

## 커뮤니티에 참여하세요

*   [GitHub Discussions](https://github.com/NVIDIA-AI-IOT/jetbot/discussions): 질문하고 JetBot 관련 주제를 토론하세요.
*   [GitHub Issues](https://github.com/NVIDIA-AI-IOT/jetbot/issues): 버그를 보고하거나 개선 사항을 제안하세요.
*   [Jetson Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70): 프로젝트를 공유하거나 질문하세요.

## 기여하기

[Contributing](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/contributing.md) 가이드라인을 참고하여 JetBot 프로젝트에 기여할 수 있습니다.

## 추가 자료

*   [3D Printing](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/3d_printing.md)
*   [Wi-Fi setup](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/wifi_setup.md)
*   [Docker Tips](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/docker_tips.md)
*   [Changes](https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/docs/changes.md)

## 라이선스

JetBot은 MIT 라이선스에 따라 배포됩니다.  자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
 
![스크린샷(47)](https://github.com/user-attachments/assets/4bfbae0d-72db-4e7e-a7d1-37cf0f71dd5d)

![스크린샷(50)](https://github.com/user-attachments/assets/b7032edb-6142-4f82-813b-bd6c241a3434)

![스크린샷(51)](https://github.com/user-attachments/assets/640a6971-3b6c-4e8a-95bf-e1e61a81f0ff)

![스크린샷(52)](https://github.com/user-attachments/assets/170fafa5-f6b1-42bb-9daf-d5e2cc39cc81)

![스크린샷(53)](https://github.com/user-attachments/assets/bfd2cc28-e978-4c7c-8747-c2ea6050a27d)


![스크린샷(54)](https://github.com/user-attachments/assets/2c56d425-501c-4b1f-86e5-c7ea2ff6280b)

![스크린샷(55)](https://github.com/user-attachments/assets/5e22e9ae-182c-4a77-878c-3b960904e0a1)

 ![스크린샷(56)](https://github.com/user-attachments/assets/6ada874d-9021-463b-8803-76222415aed3)

![스크린샷(57)](https://github.com/user-attachments/assets/eb386456-714d-461c-b8a8-d8d3ce107c78)

![스크린샷(58)](https://github.com/user-attachments/assets/5e15e626-8352-4c77-80f8-c9706c06a095)

![스크린샷(59)](https://github.com/user-attachments/assets/655e7708-5c61-4d1a-bb6b-30f0160eaec7)

![스크린샷(60)](https://github.com/user-attachments/assets/8281aa53-751d-45ac-bc96-3094accf3804)

 
![스크린샷(61)](https://github.com/user-attachments/assets/7a4dff39-f2e4-449e-a931-1f9523446138)



