# 브라우저 기반 강화학습 CartPole 프로젝트

이 프로젝트는 웹 브라우저에서 실행되는 CartPole 게임을 강화학습으로 훈련시키고, 브라우저 내에서 직접 AI가 플레이하도록 구현한 것입니다. Python의 `stable-baselines3`를 사용하여 에이전트를 훈련하고, WebSocket을 통해 브라우저의 게임 환경과 실시간으로 통신합니다.

## ✨ 주요 기능

- **웹 기반 시뮬레이션**: `Matter.js` 물리 엔진을 사용하여 브라우저에서 인터랙티브한 CartPole 환경을 제공합니다.
- **세 가지 실행 모드**:
    1.  **USER PLAY**: 사용자가 직접 키보드로 카트를 조작합니다.
    2.  **AI PLAY**: 훈련된 `model.onnx` 파일을 브라우저에서 직접 실행하여 AI가 플레이합니다. (`onnxruntime-web` 사용)
    3.  **TRAINING**: Python 훈련 스크립트와 연동하여 실시간으로 에이전트를 훈련합니다.
- **강화학습**: `stable-baselines3` 라이브러리의 PPO 알고리즘을 사용하여 모델을 훈련합니다.
- **실시간 연동**: Python의 `WebSocket` 서버와 브라우저의 `WebSocket` 클라이언트가 실시간으로 상태(Observation)와 행동(Action)을 주고받습니다.
- **모델 변환**: 훈련된 `stable-baselines3` 모델(`.zip`)을 브라우저에서 사용 가능한 `.onnx` 포맷으로 변환하는 스크립트를 제공합니다.
- **자동화된 훈련**: 훈련 스크립트 실행 시 자동으로 웹 서버를 구동하고 브라우저를 엽니다. 또한, 가장 최근의 체크포인트에서 훈련을 자동으로 재개합니다.

## 📂 프로젝트 구조

```
.
├── train_simple.py       # 강화학습 훈련을 시작하는 메인 스크립트
├── websocket_env.py      # 브라우저와 통신하는 WebSocket 기반 Gym 환경
├── convert_onnx.py       # SB3 모델을 ONNX 포맷으로 변환하는 스크립트
├── index.html            # 게임 화면, 시뮬레이션, UI를 담당하는 프론트엔드
├── model.onnx            # (생성됨) 브라우저 AI 플레이용 모델
└── training_logs/        # (생성됨) 훈련 로그 및 모델 체크포인트 저장
```

## ⚙️ 설치

프로젝트 실행에 필요한 Python 라이브러리들을 설치합니다.

```bash
pip install stable-baselines3[extra] torch gymnasium websockets onnx onnxruntime
```

- `stable-baselines3[extra]`: 강화학습 프레임워크 및 TensorBoard 로깅 지원
- `torch`: 신경망 모델의 백엔드
- `gymnasium`: 강화학습 환경 표준
- `websockets`: Python WebSocket 서버 구현
- `onnx`, `onnxruntime`: ONNX 모델 변환 및 검증

## 🚀 사용법

### 1. 모델 훈련하기

1.  터미널에서 아래 명령어를 실행하여 훈련을 시작합니다.
    ```bash
    python train_simple.py
    ```
2.  스크립트가 실행되면 자동으로 웹 브라우저가 열리고 `index.html`이 로드됩니다.
3.  화면 중앙의 모달 창에서 **[TRAINING]** 버튼을 클릭합니다.
4.  터미널과 브라우저의 WebSocket 상태 표시를 통해 훈련이 진행되는 것을 확인할 수 있습니다.
5.  훈련이 완료되면 `training_logs` 폴더 내부에 `final_model.zip` 파일이 저장됩니다.

### 2. 훈련된 모델을 ONNX로 변환하기

훈련된 모델을 브라우저의 'AI PLAY' 모드에서 사용하려면 `.onnx` 파일로 변환해야 합니다.

1.  훈련이 끝난 후, 터미널에서 아래 명령어를 실행합니다.
    ```bash
    python convert_onnx.py
    ```
2.  이 스크립트는 `training_logs` 폴더에서 가장 최근에 훈련된 모델 (`final_model.zip` 또는 체크포인트)을 찾아 `model.onnx` 파일로 자동 변환합니다.

### 3. 브라우저에서 AI 플레이하기

`model.onnx` 파일이 생성되었다면, 브라우저에서 AI의 플레이를 볼 수 있습니다.

1.  `index.html` 파일을 직접 열거나, `python train_simple.py`를 실행하여 브라우저를 엽니다 (터미널은 바로 종료해도 됩니다).
2.  화면 중앙의 모달 창에서 **[AI PLAY]** 버튼을 클릭합니다.
3.  AI가 `model.onnx`를 로드하여 자동으로 게임을 플레이합니다.

### 4. 직접 플레이하기

- **[USER PLAY]** 버튼을 클릭하면 키보드 방향키(←, →)로 직접 카트를 움직여 게임을 즐길 수 있습니다.
- 위/아래 방향키(↑, ↓)로 카트의 속도를 조절할 수 있습니다.
