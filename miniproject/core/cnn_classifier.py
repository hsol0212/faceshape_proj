# -*- coding: utf-8 -*-
"""
core/cnn_classifier.py   [갈래 A 담당]
=======================================
CNN 얼굴형 분류만 담당합니다. 랜드마크/AR 쪽(core/landmark_analyzer.py,
core/glasses_ar.py)과는 완전히 독립적인 파일이라 서로 안 건드리고 작업 가능합니다.

가위바위보 실습 코드(2_train_jetson.py / 4_predict_jetson.py) 구조를 재사용하되,
출력 클래스를 4(가위/바위/보/배경) -> 4(Round/Oval/Square/Rectangle, Heart 제외)로 변경.
학습은 training/prepare_dataset.py + training/train_faceshape.py 로 별도 진행해서
faceshape_weights.npz 를 만들어낸다는 전제입니다.

다른 파일(app/jetson_client.py 등)에서 쓰는 인터페이스:
    load_model(weights_dir=".", transfer=True)   ← 설정만 저장, 실제 로드는 지연됨
    predict_face_shape(face_crop_bgr, gender) -> (face_shape: str, confidence: float)
        gender는 "M"/"F" 또는 "men"/"women" 허용
이 두 함수 시그니처만 유지하면 내부 구현은 자유롭게 바꿔도 됩니다.

※ v2 변경사항: 성별별로 얼굴형 특징이 다르게 나타나서, 모델을 남/여 따로 학습하기로
   결정했습니다 (training/prepare_dataset.py, training/train_faceshape.py도 함께 변경됨).
   predict_face_shape()에 gender 인자가 추가된 게 기존과 다른 점입니다.

※ v3 변경사항: Jetson Nano에서 men+women 모델을 동시에 GPU에 올리다 OOM(메모리
   부족)이 발생해서, 지연 로딩 + 최대 1개 모델만 유지하는 방식으로 변경했습니다.
   성별을 전환하면 이전 모델은 메모리에서 비워지고 새로 로드됩니다 (조금 느려지지만
   메모리 여유가 없는 Jetson Nano에서는 이 방식이 안전합니다).
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# GPU 메모리를 한 번에 왕창 잡지 말고 필요한 만큼만 늘려가며 쓰도록 설정.
# (Jetson Nano처럼 GPU 메모리가 빠듯한 환경에서 OOM 방지에 도움됨)
try:
    for _gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception as _e:
    print(f"[cnn_classifier] GPU 메모리 growth 설정 실패 (무시하고 계속): {_e}")

VER, HOR = 224, 224
# Heart 제외 4클래스 — training/prepare_dataset.py와 순서 반드시 동일해야 함
FACE_SHAPE_CLASSES = ["Oval", "Rectangle", "Round", "Square"]

_model = None  # 지연 로딩 (모듈 임포트 시점에 바로 로드하지 않음)


def _build_model():
    """가위바위보 코드와 동일한 얕은 CNN 구조 (from scratch).
    주의: 얼굴형 데이터는 카테고리 간 차이가 미묘해서 처음부터 학습하면 정확도가
    낮게 나올 수 있습니다 — 기본은 build_transfer_model() 사용을 권장합니다.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(VER, HOR, 3), name='conv_1'),
        MaxPooling2D((2, 2), name='pool_1'),
        Conv2D(64, (3, 3), activation='relu', name='conv_2'),
        MaxPooling2D((2, 2), name='pool_2'),
        Conv2D(64, (3, 3), activation='relu', name='conv_3'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(len(FACE_SHAPE_CLASSES), activation='softmax', name='dense_out'),
    ])
    return model


def build_transfer_model():
    """MobileNetV2 전이학습 버전 (기본값).
    training/train_faceshape.py와 레이어 구성이 정확히 동일해야 가중치가 맞게 들어갑니다.
    (Sequential 순서: Rescaling(1/255) → MobileNetV2 → GAP → Dense(128) → Dropout(0.3) → Dense(4, softmax))
    ※ Rescaling이 0~255 -> 0~1 변환을 대신하므로, predict_face_shape()에서 따로
      /255 정규화를 하지 않습니다 (이중으로 나누면 안 됨 — 아래 주석 참고).
    """
    from tensorflow.keras.applications import MobileNetV2
    # TF 버전에 따라 Rescaling 위치가 다름 (PC의 TF 2.15는 tensorflow.keras.layers,
    # Jetson의 TF 2.4.1은 tensorflow.keras.layers.experimental.preprocessing에 있음)
    try:
        from tensorflow.keras.layers import Rescaling
    except ImportError:
        from tensorflow.keras.layers.experimental.preprocessing import Rescaling

    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(VER, HOR, 3))
    # weights=None: 어차피 학습된 가중치를 아래에서 파일로 덮어씌우므로 imagenet 재다운로드 불필요
    model = Sequential([
        Rescaling(1.0 / 255, input_shape=(VER, HOR, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(FACE_SHAPE_CLASSES), activation='softmax'),
    ])
    return model


_models = {}       # {"men": model, "women": model} — 실제로 로드된 것만 들어있음
_weights_dir = "."
_transfer = True


def load_model(weights_dir=".", transfer=True):
    """설정만 저장해둡니다. 실제 가중치 로드는 predict_face_shape()가 처음
    해당 성별로 호출될 때 그제서야 이뤄집니다 (지연 로딩).

    이렇게 바꾼 이유: men/women 모델을 시작하자마자 둘 다 GPU에 올리면
    Jetson Nano처럼 GPU 메모리가 빠듯한 환경에서 OOM(메모리 부족)이 날 수
    있습니다. 실제로 쓰는 성별 하나만 메모리에 있으면 되므로, 필요할 때만
    불러오는 방식으로 메모리 사용량을 절반으로 줄입니다.

    weights_dir 안에 faceshape_weights_men.npz, faceshape_weights_women.npz가
    있어야 합니다 (training/train_faceshape.py의 산출물).
    """
    global _weights_dir, _transfer
    _weights_dir = weights_dir
    _transfer = transfer
    print("[cnn_classifier] 초기화 완료 (모델은 처음 예측 요청 시 필요한 성별만 로드됩니다)")


def _ensure_loaded(gender_key):
    if gender_key in _models:
        return
    if _models:
        # 이미 다른 성별 모델이 메모리에 있으면 완전히 비우고 새로 로드
        # (Jetson Nano처럼 메모리가 아주 빠듯하면 2개를 동시에 못 들고 있음 —
        #  최대 1개만 유지해서 메모리를 최소로 씀. PC처럼 여유 있는 환경에서는
        #  성별 왔다갔다 할 때마다 다시 로드하느라 조금 느려질 수 있는 게 트레이드오프)
        tf.keras.backend.clear_session()
        _models.clear()
        print("[cnn_classifier] 메모리 확보를 위해 이전 모델을 비웠습니다.")

    weights_path = os.path.join(_weights_dir, f"faceshape_weights_{gender_key}.npz")
    model = build_transfer_model() if _transfer else _build_model()
    npz_file = np.load(weights_path)
    weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
    model.set_weights(weights)
    _models[gender_key] = model
    print(f"[cnn_classifier] {gender_key} 얼굴형 모델 가중치 로드 완료 "
          f"({'transfer' if _transfer else 'scratch'}, 지연 로딩)")


def predict_face_shape(face_crop_bgr, gender, use_tta=True):
    """crop된 얼굴 이미지 + 성별("M"/"F" 또는 "men"/"women") -> (얼굴형 문자열, confidence 0~100)

    use_tta=True(기본값): 원본 + 좌우반전 두 장을 같이 넣어서 예측을 평균냅니다
    (Test-Time Augmentation). 재학습 없이 정확도를 조금 더 끌어올리는 방법이라
    기본으로 켜뒀습니다. Jetson에서 속도가 부족하면 False로 끄면 원래처럼 1장만 씁니다.
    """
    gender_key = {"M": "men", "F": "women", "men": "men", "women": "women"}.get(gender)
    if gender_key is None:
        raise ValueError(f"gender는 'M'/'F'/'men'/'women' 중 하나여야 합니다: {gender}")

    _ensure_loaded(gender_key)
    model = _models[gender_key]
    img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (HOR, VER), interpolation=cv2.INTER_CUBIC)
    # 주의: 여기서 /255 하지 않습니다 — 모델 맨 앞의 Rescaling(1/255) 레이어가 대신 처리합니다.
    x = img.astype("float32")

    if use_tta:
        x_flipped = np.flip(x, axis=1)  # 좌우반전
        batch = np.stack([x, x_flipped], axis=0)  # (2, H, W, 3)
        preds = model.predict(batch, verbose=0)
        pred = preds.mean(axis=0, keepdims=True)  # 두 예측의 평균
    else:
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x, verbose=0)

    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx] * 100)
    return FACE_SHAPE_CLASSES[idx], confidence
