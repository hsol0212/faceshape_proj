# -*- coding: utf-8 -*-
"""
train_faceshape.py
===================
수업 템플릿(2_1_train.전이학습_검증.py)을 얼굴형 5클래스용으로 개조한 버전.
PC/Colab에서 실행 (Jetson에서 직접 학습하지 않음 — 기존 가위바위보 실습과 동일한 흐름).

산출물:
  - my_cnn_model_faceshape.h5   (PC에서 검증용으로 다시 불러볼 모델)
  - faceshape_weights.npz       (Jetson으로 옮길 가중치만 추출한 파일)
    → 이 파일을 faceshape_project/faceshape_weights.npz 자리에 넣으면 core.py가 그대로 사용

core.py의 build_transfer_model()과 레이어 구성이 정확히 같아야 가중치가 맞게 들어갑니다.
구조를 바꾸면 core.py도 같이 바꿔야 합니다 (Sequential 순서: MobileNetV2 → GAP →
Dense(128) → Dropout(0.3) → Dense(5, softmax)).
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

VER, HOR = 224, 224
NUM_CLASSES = 4
CLASS_NAMES = ["Oval", "Rectangle", "Round", "Square"]  # prepare_dataset.py와 순서 동일해야 함

DATASET_NPZ = r".\dataset_faceshape.npz"  # prepare_dataset.py의 OUTPUT_NPZ와 동일 경로

# ------------------------------------------------------------------
# 1. 데이터셋 불러오기
# ------------------------------------------------------------------
data = np.load(DATASET_NPZ)
x_train, x_test = data["x_train"], data["x_test"]
y_train, y_test = data["y_train"], data["y_test"]

print(f"학습셋: {x_train.shape[0]}장 / 테스트셋: {x_test.shape[0]}장")

x_train_encoded = x_train.astype("float32") / 255.0
x_test_encoded = x_test.astype("float32") / 255.0

# 원본(uint8) 배열은 이제 필요 없으므로 메모리에서 비워줌
# (prepare_dataset.py 단계에서 겪었던 MemoryError가 여기서도 날 수 있어서 미리 방지)
del x_train, x_test, data
gc.collect()

y_train_encoded = utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_encoded = utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# ------------------------------------------------------------------
# 2. MobileNetV2 기반 전이학습 모델
#    ※ core.py의 build_transfer_model()과 레이어 구성이 반드시 동일해야 함
# ------------------------------------------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(VER, HOR, 3))
base_model.trainable = False  # 1차: 사전학습 가중치 고정 (특징 추출기로만 사용)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax"),
])
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

# ------------------------------------------------------------------
# 3. 학습
# ------------------------------------------------------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath="my_cnn_model_faceshape.h5", monitor="val_loss", save_best_only=True, verbose=1
)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start = time.time()
        print("\n=== 학습 시작 ===")

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start
        print(f"\n총 학습 소요 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")


history = model.fit(
    x_train_encoded, y_train_encoded,
    epochs=50,
    batch_size=16,
    validation_data=(x_test_encoded, y_test_encoded),
    callbacks=[early_stopping, checkpoint, TimeHistory()],
)

# ------------------------------------------------------------------
# 4. 평가
# ------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test_encoded, y_test_encoded, verbose=0)
print(f"\nFinal Best Test Accuracy: {test_acc:.4f}")

y_pred = model.predict(x_test_encoded, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n[상세 분류 성능 리포트]")
print(classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES))
print("[혼동 행렬]")
print(confusion_matrix(y_test, y_pred_classes))

# 학습 곡선
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["acc"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_acc"], label="Validation Accuracy", marker="x")
plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Validation Loss", marker="x")
plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")
print("학습 곡선을 training_curve.png로 저장했습니다.")

# ------------------------------------------------------------------
# 5. Jetson 이식용 가중치 추출 (핵심 — 이걸 빼먹으면 Jetson에서 못 씀)
# ------------------------------------------------------------------
weights = model.get_weights()
np.savez("faceshape_weights.npz", *weights)
print(f"\nJetson 이식용 가중치 저장 완료: faceshape_weights.npz ({len(weights)}개 배열)")
print("이 파일을 faceshape_project/ 폴더의 faceshape_weights.npz 자리에 넣으면 됩니다.")
