# -*- coding: utf-8 -*-
"""
core/cnn_classifier.py   [갈래 A 담당]
=======================================
CNN 얼굴형 분류만 담당합니다. 랜드마크/AR 쪽(core/landmark_analyzer.py,
core/glasses_ar.py)과는 완전히 독립적인 파일이라 서로 안 건드리고 작업 가능합니다.

가위바위보 실습 코드(2_train_jetson.py / 4_predict_jetson.py) 구조를 재사용하되,
출력 클래스를 4(가위/바위/보/배경) -> 4(Oblong/Oval/Round/Square, Heart 제외)로 변경.
학습은 training/prepare_dataset.py + training/train_faceshape.py 로 별도 진행해서
faceshape_weights.npz 를 만들어낸다는 전제입니다.

다른 파일(app/jetson_client.py 등)에서 쓰는 인터페이스:
    load_model(weights_path, transfer=True)
    predict_face_shape(face_crop_bgr) -> (face_shape: str, confidence: float)
이 두 함수 시그니처만 유지하면 내부 구현은 자유롭게 바꿔도 됩니다.
"""

import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

VER, HOR = 224, 224
# Heart 제외 4클래스 — training/prepare_dataset.py와 순서 반드시 동일해야 함
FACE_SHAPE_CLASSES = ["Oblong", "Oval", "Round", "Square"]

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
    (Sequential 순서: MobileNetV2 → GAP → Dense(128) → Dropout(0.3) → Dense(4, softmax))
    """
    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(VER, HOR, 3))
    # weights=None: 어차피 학습된 가중치를 아래에서 파일로 덮어씌우므로 imagenet 재다운로드 불필요
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(FACE_SHAPE_CLASSES), activation='softmax'),
    ])
    return model


def load_model(weights_path="faceshape_weights.npz", transfer=True):
    global _model
    _model = build_transfer_model() if transfer else _build_model()
    npz_file = np.load(weights_path)
    weights = [npz_file[f'arr_{i}'] for i in range(len(npz_file.files))]
    _model.set_weights(weights)
    print(f"[cnn_classifier] 얼굴형 모델 가중치 로드 완료 ({'transfer' if transfer else 'scratch'})")


def predict_face_shape(face_crop_bgr):
    """crop된 얼굴 이미지 -> (얼굴형 문자열, confidence 0~100)"""
    if _model is None:
        raise RuntimeError("load_model()을 먼저 호출하세요.")

    img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (HOR, VER), interpolation=cv2.INTER_CUBIC)
    x = img.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    pred = _model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx] * 100)
    return FACE_SHAPE_CLASSES[idx], confidence
