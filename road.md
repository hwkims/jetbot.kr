```markdown
# JetBot 도로 주행 프로젝트: 데이터 수집부터 실시간 데모까지 (TensorRT 최적화 포함)

## 1. 프로젝트 개요

본 문서는 NVIDIA JetBot을 사용하여 카메라 이미지를 기반으로 도로(트랙)를 따라 자율적으로 주행하는 프로젝트의 전체 과정을 설명합니다. 이 프로젝트는 다음 단계로 구성됩니다.

1.  **데이터 수집:** JetBot의 카메라를 이용하여 주행 경로 이미지를 수집하고, 각 이미지에서 로봇이 나아가야 할 목표 지점(x, y 좌표)을 라벨링합니다.
2.  **모델 학습:** 수집된 데이터를 사용하여 이미지로부터 목표 지점의 (x, y) 좌표를 예측하는 딥러닝 회귀 모델(ResNet-18 기반)을 학습시킵니다.
3.  **TensorRT 최적화:** 학습된 PyTorch 모델을 TensorRT를 사용하여 최적화하여 JetBot과 같은 엣지 디바이스에서 더 빠른 추론 속도를 확보합니다.
4.  **실시간 데모:** 최적화된 모델을 JetBot에 배포하여 실시간 카메라 입력을 통해 도로 주행을 수행합니다.

## 2. 데이터 수집 (Data Collection)

도로 주행 모델을 학습시키기 위한 첫 단계는 JetBot이 실제 주행할 환경에서 데이터를 수집하는 것입니다.

### 2.1. 목표

*   JetBot의 카메라로 트랙 이미지를 캡처합니다.
*   각 이미지에서 로봇이 향해야 할 목표 지점의 (x, y) 픽셀 좌표를 기록합니다.
*   다양한 위치와 각도에서 데이터를 수집하여 모델의 강건성(Robustness)을 높입니다.

### 2.2. 데이터 라벨링 가이드

1.  카메라의 실시간 피드를 관찰합니다.
2.  로봇이 따라가야 할 경로를 상상합니다 (트랙 중앙 또는 특정 라인).
3.  로봇이 해당 지점으로 직진했을 때 경로를 벗어나지 않을 **가장 먼 지점**을 선택하여 클릭합니다.
    *   직선 경로에서는 멀리 있는 지점을 선택할 수 있습니다.
    *   급격한 커브에서는 로봇이 경로를 벗어나지 않도록 더 가까운 지점을 선택해야 합니다.

### 2.3. 구현 코드

필요한 라이브러리를 가져오고 카메라 및 위젯을 설정합니다. `jupyter_clickable_image_widget`을 사용하여 이미지 위에 직접 클릭하여 좌표를 얻습니다.

```python
# IPython Libraries for display and widgets
import ipywidgets
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display

# Camera and Motor Interface for JetBot
from jetbot import Robot, Camera, bgr8_to_jpeg

# Basic Python packages for image annotation
from uuid import uuid1
import os
import json
import glob
import datetime
import numpy as np
import cv2
import time

# Clickable image widget
from jupyter_clickable_image_widget import ClickableImageWidget

# 데이터셋 저장 디렉토리
DATASET_DIR = 'dataset_xy'

# 디렉토리 생성 (이미 존재하면 오류 무시)
try:
    os.makedirs(DATASET_DIR)
except FileExistsError:
    print('디렉토리가 이미 존재하여 새로 생성하지 않았습니다.')

# 카메라 초기화 (모델 입력 크기에 맞춰 224x224 설정)
camera = Camera(width=224, height=224)

# 이미지 위젯 생성 (클릭 가능)
camera_widget = ClickableImageWidget(width=camera.width, height=camera.height)
# 마지막 저장된 스냅샷 표시용 위젯
snapshot_widget = ipywidgets.Image(width=camera.width, height=camera.height)
# 카메라 출력을 camera_widget에 연결 (BGR -> JPEG 변환)
traitlets.dlink((camera, 'value'), (camera_widget, 'value'), transform=bgr8_to_jpeg)

# 저장된 이미지 개수 표시 위젯
count_widget = ipywidgets.IntText(description='count')
# 초기 이미지 개수 업데이트
count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

# 이미지 클릭 시 실행될 콜백 함수
def save_snapshot(_, content, msg):
    if content['event'] == 'click':
        data = content['eventData']
        x = data['offsetX'] # 클릭된 x 좌표
        y = data['offsetY'] # 클릭된 y 좌표

        # 이미지를 디스크에 저장
        # 파일명 형식: xy_<x좌표>_<y좌표>_<uuid>.jpg
        uuid_str = str(uuid1())
        image_path = os.path.join(DATASET_DIR, f'xy_{x:03d}_{y:03d}_{uuid_str}.jpg')
        with open(image_path, 'wb') as f:
            f.write(camera_widget.value) # JPEG 형식으로 저장

        # 저장된 스냅샷 미리보기 (클릭 지점에 녹색 원 표시)
        snapshot = camera.value.copy() # 현재 카메라 프레임 복사 (BGR 형식)
        snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3) # 원 그리기
        snapshot_widget.value = bgr8_to_jpeg(snapshot) # JPEG로 변환하여 위젯에 표시

        # 이미지 개수 업데이트
        count_widget.value = len(glob.glob(os.path.join(DATASET_DIR, '*.jpg')))

# camera_widget에서 클릭 이벤트 발생 시 save_snapshot 함수 호출
camera_widget.on_msg(save_snapshot)

# 위젯들을 VBox로 묶어 표시
data_collection_widget = ipywidgets.VBox([
    ipywidgets.HBox([camera_widget, snapshot_widget]), # 카메라 피드와 스냅샷
    count_widget # 이미지 개수
])

# 데이터 수집 위젯 표시
display(data_collection_widget)
```

### 2.4. 데이터 확인

수집 후 `dataset_xy` 폴더에는 `xy_<x>_<y>_<uuid>.jpg` 형식의 파일들이 저장됩니다. 파일명에 포함된 x, y 좌표는 이후 모델 학습 시 라벨로 사용됩니다.

## 3. 모델 학습 (Train Model)

수집된 데이터를 이용하여 도로 주행을 위한 딥러닝 모델을 학습시킵니다.

### 3.1. 목표

*   입력 이미지로부터 목표 지점의 정규화된 (x, y) 좌표를 예측하는 회귀 모델을 학습합니다.
*   PyTorch 프레임워크와 ResNet-18 아키텍처(전이 학습 활용)를 사용합니다.

### 3.2. 데이터셋 준비

#### 3.2.1. 커스텀 데이터셋 클래스 (`XYDataset`)

이미지를 로드하고 파일명에서 x, y 좌표를 파싱하며, 필요한 전처리(크기 조정, 텐서 변환, 정규화) 및 데이터 증강(Color Jitter, Random Horizontal Flip)을 수행하는 `Dataset` 클래스를 정의합니다.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

# 파일명에서 x 좌표 추출 및 정규화 (-1 ~ 1 범위)
def get_x(path, width):
    """Gets the x value from the image filename"""
    return (float(int(path.split("_")[1])) - width/2) / (width/2)

# 파일명에서 y 좌표 추출 및 정규화 (-1 ~ 1 범위)
def get_y(path, height):
    """Gets the y value from the image filename"""
    return (float(int(path.split("_")[2])) - height/2) / (height/2)

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        # 데이터 증강을 위한 Color Jitter 설정
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # 이미지 로드
        image = PIL.Image.open(image_path)
        width, height = image.size
        # 파일명에서 x, y 좌표 추출 및 정규화
        x = float(get_x(os.path.basename(image_path), width))
        y = float(get_y(os.path.basename(image_path), height))

        # Random Horizontal Flip (데이터 증강)
        if self.random_hflips and float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x # x 좌표 반전

        # 데이터 증강: Color Jitter
        image = self.color_jitter(image)
        # 이미지 크기 조정 (ResNet 입력 크기)
        image = transforms.functional.resize(image, (224, 224))
        # 이미지를 Tensor로 변환
        image = transforms.functional.to_tensor(image)
        # OpenCV (BGR) 순서로 변경 (선택 사항, 원본 코드 기준)
        # image = image.numpy()[::-1].copy()
        # image = torch.from_numpy(image)
        # 이미지 정규화 (ImageNet 평균 및 표준편차 사용)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 이미지와 라벨(x, y 좌표) 반환
        return image, torch.tensor([x, y]).float()

# 데이터셋 인스턴스 생성
dataset = XYDataset('dataset_xy', random_hflips=False) # 필요시 random_hflips=True 설정
```

#### 3.2.2. 데이터 분할 및 로더 생성

데이터셋을 훈련(Train) 세트와 테스트(Test) 세트로 분할하고, 배치(Batch) 단위로 데이터를 로드하는 `DataLoader`를 생성합니다.

```python
# 테스트 데이터 비율 설정 (예: 10%)
test_percent = 0.1
num_test = int(test_percent * len(dataset))
num_train = len(dataset) - num_test

# 데이터셋 분할
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

# 데이터 로더 생성
BATCH_SIZE = 8 # GPU 메모리에 따라 조절
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # 훈련 데이터는 섞음
    num_workers=0 # Jetson Nano에서는 0 권장
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # 테스트 데이터는 섞지 않음
    num_workers=0
)
```

### 3.3. 모델 정의 (ResNet-18)

ImageNet으로 사전 학습된 ResNet-18 모델을 로드하고, 마지막 Fully Connected Layer (fc)를 교체하여 (x, y) 좌표 2개를 출력하도록 수정합니다.

```python
# 사전 학습된 ResNet-18 모델 로드
model = models.resnet18(pretrained=True)

# 마지막 FC 레이어 교체 (출력 크기를 2로 설정: x, y 좌표)
model.fc = torch.nn.Linear(512, 2)

# 모델을 GPU로 이동
device = torch.device('cuda')
model = model.to(device)
```

### 3.4. 모델 학습

Adam 옵티마이저와 Mean Squared Error (MSE) 손실 함수를 사용하여 모델을 학습시킵니다. 각 에포크마다 훈련 및 테스트 손실을 계산하고, 테스트 손실이 가장 낮은 모델을 저장합니다.

```python
# 학습 파라미터 설정
NUM_EPOCHS = 70 # 총 에포크 수
BEST_MODEL_PATH = 'best_steering_model_xy.pth' # 최고 성능 모델 저장 경로
best_loss = 1e9 # 최고 성능(최저 손실) 기록 변수

# 옵티마이저 정의 (Adam 사용)
optimizer = optim.Adam(model.parameters())

# 학습 루프
for epoch in range(NUM_EPOCHS):
    # 훈련 모드
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        # 데이터를 GPU로 이동
        images = images.to(device)
        labels = labels.to(device)
        # 옵티마이저 그래디언트 초기화
        optimizer.zero_grad()
        # 모델 예측
        outputs = model(images)
        # 손실 계산 (MSE Loss)
        loss = F.mse_loss(outputs, labels)
        # 훈련 손실 누적
        train_loss += loss.item()
        # 역전파
        loss.backward()
        # 옵티마이저 스텝 (가중치 업데이트)
        optimizer.step()
    # 평균 훈련 손실 계산
    train_loss /= len(train_loader)

    # 평가 모드
    model.eval()
    test_loss = 0.0
    with torch.no_grad(): # 그래디언트 계산 비활성화
        for images, labels in iter(test_loader):
            # 데이터를 GPU로 이동
            images = images.to(device)
            labels = labels.to(device)
            # 모델 예측
            outputs = model(images)
            # 손실 계산
            loss = F.mse_loss(outputs, labels)
            # 테스트 손실 누적
            test_loss += loss.item()
    # 평균 테스트 손실 계산
    test_loss /= len(test_loader)

    # 에포크별 손실 출력
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

    # 최고 성능 모델 저장
    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f'    Best model saved with test loss: {test_loss:.6f}')
        best_loss = test_loss

print("학습 완료!")
```

### 3.5. 학습 결과 (예시)

학습 중 출력되는 로그는 다음과 같은 형식을 가집니다. 테스트 손실이 개선될 때마다 모델 파일(`best_steering_model_xy.pth`)이 갱신됩니다.

```
Epoch 1/70, Train Loss: 0.297352, Test Loss: 0.083579
    Best model saved with test loss: 0.083579
Epoch 2/70, Train Loss: 0.054634, Test Loss: 0.064844
    Best model saved with test loss: 0.064844
Epoch 3/70, Train Loss: 0.064151, Test Loss: 0.057499
    Best model saved with test loss: 0.057499
... (중략) ...
Epoch 52/70, Train Loss: 0.042623, Test Loss: 0.021270
    Best model saved with test loss: 0.021270
... (중략) ...
Epoch 65/70, Train Loss: 0.041549, Test Loss: 0.023550
Epoch 66/70, Train Loss: 0.042796, Test Loss: 0.047188
Epoch 67/70, Train Loss: 0.043570, Test Loss: 0.028584
Epoch 68/70, Train Loss: 0.036441, Test Loss: 0.033807
Epoch 69/70, Train Loss: 0.035894, Test Loss: 0.051173
Epoch 70/70, Train Loss: 0.039792, Test Loss: 0.039690
학습 완료!
```

학습이 완료되면 `best_steering_model_xy.pth` 파일이 생성됩니다. 이 파일은 학습된 모델의 가중치를 담고 있습니다.

## 4. TensorRT 최적화 (Build TensorRT model)

JetBot과 같은 임베디드 환경에서 더 빠른 추론 속도를 얻기 위해 학습된 PyTorch 모델을 TensorRT로 변환하고 최적화합니다.

### 4.1. 목표

*   `best_steering_model_xy.pth` 모델을 TensorRT 엔진으로 변환합니다.
*   `torch2trt` 라이브러리를 사용하여 변환 과정을 간소화합니다.
*   최적화된 모델을 `best_steering_model_xy_trt.pth` 파일로 저장합니다.

### 4.2. `torch2trt` 설치 (필요시)

`torch2trt`가 설치되어 있지 않다면, JetBot 터미널에서 다음 명령어를 실행하여 설치합니다.

```bash
cd $HOME
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install
```

### 4.3. 모델 로드 및 변환

먼저 학습된 PyTorch 모델(`best_steering_model_xy.pth`)을 로드한 다음, `torch2trt`를 사용하여 TensorRT 모델로 변환합니다.

```python
import torchvision
import torch
from torch2trt import torch2trt, TRTModule

# 모델 구조 정의 (학습 시와 동일하게)
model = torchvision.models.resnet18(pretrained=False) # 사전 학습 가중치는 로드 안 함
model.fc = torch.nn.Linear(512, 2)

# 학습된 가중치 로드
model.load_state_dict(torch.load('best_steering_model_xy.pth'))

# 모델을 GPU로 이동하고 평가 모드로 설정
device = torch.device('cuda')
model = model.cuda().eval().half() # half()는 FP16 사용

# TensorRT 변환을 위한 더미 입력 데이터 생성
# (배치 크기, 채널, 높이, 너비) 형식, 모델 입력과 동일하게
data = torch.zeros((1, 3, 224, 224)).cuda().half()

# torch2trt를 사용하여 모델 변환 (FP16 모드 활성화)
# 이 과정은 몇 분 정도 소요될 수 있습니다.
model_trt = torch2trt(model, [data], fp16_mode=True)

# 최적화된 모델 저장
torch.save(model_trt.state_dict(), 'best_steering_model_xy_trt.pth')

print("TensorRT 모델 변환 및 저장 완료: best_steering_model_xy_trt.pth")
```

이제 추론 속도가 향상된 `best_steering_model_xy_trt.pth` 파일이 생성되었습니다.

## 5. 실시간 데모 (Live demo - TensorRT)

최적화된 TensorRT 모델을 사용하여 JetBot이 실시간으로 도로를 주행하도록 합니다.

### 5.1. 목표

*   TensorRT 모델(`best_steering_model_xy_trt.pth`)을 로드합니다.
*   실시간 카메라 입력을 받아 모델 추론을 수행합니다.
*   모델의 예측값(x, y 좌표)을 사용하여 로봇의 조향 및 속도를 제어합니다.
*   사용자가 주행 파라미터(속도, 조향 게인 등)를 조절할 수 있는 인터페이스를 제공합니다.

### 5.2. TensorRT 모델 로드

`TRTModule`을 사용하여 저장된 TensorRT 모델 상태를 로드합니다.

```python
import torch
from torch2trt import TRTModule

# TRTModule 인스턴스 생성
model_trt = TRTModule()

# 저장된 TensorRT 모델 상태 로드
model_trt.load_state_dict(torch.load('best_steering_model_xy_trt.pth'))

# GPU 설정
device = torch.device('cuda')

print("TensorRT 모델 로드 완료.")
```

### 5.3. 전처리 함수 정의

카메라에서 얻은 이미지(일반적으로 NumPy 배열, HWC, BGR, 0-255 범위)를 모델 입력 형식(PyTorch Tensor, CHW, RGB, 정규화, FP16, 배치 차원 추가)으로 변환하는 함수를 정의합니다.

```python
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

# ImageNet 정규화 파라미터 (FP16으로 GPU에 올림)
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    # NumPy 배열(BGR)을 PIL 이미지(RGB)로 변환
    image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # PIL 이미지를 Tensor로 변환 (CHW, 0-1 범위)
    image = transforms.functional.to_tensor(image).to(device).half() # FP16 사용
    # 정규화 수행 (inplace 연산)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    # 배치 차원 추가 (1, C, H, W)
    return image[None, ...]
```

### 5.4. 카메라 및 로봇 인터페이스 설정

카메라, 로봇 객체를 초기화하고, 카메라 피드를 표시할 위젯을 설정합니다.

```python
from IPython.display import display
import ipywidgets
import traitlets
from jetbot import Camera, bgr8_to_jpeg, Robot

# 카메라 초기화 (데이터 수집 시와 동일한 크기 권장)
camera = Camera(width=224, height=224)

# 카메라 피드 표시용 이미지 위젯
image_widget = ipywidgets.Image(width=camera.width, height=camera.height)

# 카메라 출력을 이미지 위젯에 연결 (BGR -> JPEG 변환)
traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

# 로봇 객체 초기화
robot = Robot()

# 카메라 피드 표시
display(image_widget)
```

### 5.5. 제어 파라미터 슬라이더

주행 성능을 미세 조정하기 위한 슬라이더 위젯을 생성합니다.

*   `speed_gain_slider`: 기본 속도 조절 (0 ~ 1)
*   `steering_gain_slider`: 조향 민감도 (P 제어 게인) 조절 (값이 크면 더 민감하게 반응)
*   `steering_dgain_slider`: 조향 안정성 (D 제어 게인) 조절 (값이 크면 변화에 더 빠르게 반응, 진동 억제 도움)
*   `steering_bias_slider`: 조향 편향 조절 (로봇이 한쪽으로 쏠릴 때 보정)

```python
# 속도 게인 슬라이더
speed_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.15, description='speed gain')
# 조향 P 게인 슬라이더
steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2, description='steering gain')
# 조향 D 게인 슬라이더
steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd')
# 조향 편향 슬라이더
steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias')

# 슬라이더 표시
display(speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider)
```

### 5.6. 예측값 및 조향 시각화 슬라이더 (선택 사항)

모델의 예측값(x, y)과 계산된 조향 값을 시각적으로 확인하기 위한 슬라이더를 추가합니다.

```python
# 예측된 x 좌표 표시
x_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='x', disabled=True)
# 예측된 y 좌표 표시 (정규화된 값이 아닌, 주행 로직에 사용되는 값 표시)
y_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='y', disabled=True)
# 최종 조향 값 표시
steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering', disabled=True)
# 현재 설정된 속도 값 표시
speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed', disabled=True)

# 슬라이더 표시
display(ipywidgets.HBox([y_slider, speed_slider]))
display(x_slider, steering_slider)
```

### 5.7. 실행 함수 (`execute`)

카메라에서 새로운 프레임이 들어올 때마다 호출될 메인 함수를 정의합니다. 이 함수는 다음 작업을 수행합니다.

1.  이미지 전처리
2.  TensorRT 모델 추론 수행하여 (x, y) 좌표 예측
3.  예측된 (x, y) 좌표를 사용하여 목표 각도(`angle`) 계산
4.  PD 제어 로직을 적용하여 최종 조향 값(`pid`) 계산
5.  계산된 속도와 조향 값으로 로봇 모터 제어

```python
import numpy as np

# PD 제어를 위한 변수 초기화
angle = 0.0
angle_last = 0.0

# 카메라 프레임 변경 시 실행될 함수
def execute(change):
    global angle, angle_last # 이전 각도 값을 유지하기 위해 global 사용

    # 새 카메라 프레임 가져오기
    image = change['new']

    # 이미지 전처리 및 모델 추론
    xy = model_trt(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0] # 예측된 x 좌표 (-1 ~ 1)
    # y 좌표 변환 (원본 코드 기준, 값이 작을수록 멀리 있는 것으로 해석)
    # 예측된 y 값(xy[1])은 -1(아래) ~ 1(위) 범위.
    # (0.5 - xy[1]) / 2.0 계산: y값이 1(위)이면 0에 가깝고, -1(아래)이면 0.5에 가까워짐
    y = (0.5 - xy[1]) / 2.0

    # 시각화 슬라이더 업데이트
    x_slider.value = x
    y_slider.value = y
    speed_slider.value = speed_gain_slider.value

    # 목표 각도 계산 (x: 좌우 편차, y: 거리 가중치 역할)
    # arctan2(x, y)는 y축(전방) 기준 x 방향으로의 각도를 라디안 단위로 반환
    angle = np.arctan2(x, y)

    # PD 제어 계산
    # P 제어: 현재 각도 오차 * P 게인
    # D 제어: 각도 변화량 * D 게인
    pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value

    # 현재 각도를 다음 계산을 위해 저장
    angle_last = angle

    # 최종 조향 값 계산 (PD 제어 값 + 조향 편향)
    steering_value = pid + steering_bias_slider.value
    steering_slider.value = steering_value # 시각화 슬라이더 업데이트

    # 모터 속도 계산
    # 기본 속도에서 조향 값을 더하거나 빼서 좌우 모터 속도 차등 적용
    # 값 범위를 0.0 ~ 1.0 사이로 제한 (clamping)
    left_motor_value = max(min(speed_slider.value + steering_value, 1.0), 0.0)
    right_motor_value = max(min(speed_slider.value - steering_value, 1.0), 0.0)

    # 로봇 모터 구동
    robot.left_motor.value = left_motor_value
    robot.right_motor.value = right_motor_value

# 초기 실행 (카메라 값이 로드되면 한번 실행)
# execute({'new': camera.value}) # observe가 설정되면 자동으로 호출되므로 주석 처리 가능

# 카메라의 'value' 속성이 변경될 때마다 execute 함수를 호출하도록 설정
camera.observe(execute, names='value')

print("카메라 observe 설정 완료. 로봇이 주행을 시작할 수 있습니다.")
print("주의: 로봇이 움직일 수 있으니 주변 공간을 확보하고 트랙 위에 올려놓으세요.")
```

### 5.8. 주행 시작 및 중지

위 코드를 실행하면 JetBot은 카메라 프레임이 업데이트될 때마다 `execute` 함수를 호출하여 주행을 시작합니다. 주행 중 슬라이더를 조절하여 로봇의 움직임을 미세 조정할 수 있습니다.

주행을 멈추려면 다음 코드를 실행하여 `observe` 연결을 해제합니다.

```python
# 카메라 observe 해제
camera.unobserve(execute, names='value')

# 로봇 모터 정지 (선택 사항)
time.sleep(0.1) # 잠시 대기 후 정지
robot.stop()

print("카메라 observe 해제 및 로봇 정지 완료.")
```

## 6. 결론

본 문서는 JetBot을 이용한 도로 주행 프로젝트의 전체 과정을 다루었습니다. 데이터 수집, PyTorch 기반 모델 학습, TensorRT 최적화, 그리고 실시간 데모 구현 단계를 통해 JetBot이 카메라 이미지만으로 트랙을 따라 주행하는 기능을 구현할 수 있음을 보였습니다. 특히 TensorRT 최적화를 통해 임베디드 환경에서의 실시간 성능을 확보하는 것이 중요합니다. 사용자는 제공된 코드와 설명을 바탕으로 자신만의 주행 환경에 맞춰 데이터를 수집하고 모델을 개선하여 더 나은 주행 성능을 달성할 수 있습니다.
```
