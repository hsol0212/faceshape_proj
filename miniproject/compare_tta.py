# -*- coding: utf-8 -*-
"""
compare_tta.py
==============
TTA(use_tta=True) 적용 전/후 정확도를 "테스트셋 전체"로 비교합니다.
verify_weights.py는 10장짜리 샘플이라 우연에 좌우되기 쉬운데, 이건 전체를 다 돌려서
진짜 효과가 있는지 확인합니다.

실행 위치: faceshape_project/training/ 안에서 실행
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core import cnn_classifier


def evaluate(gender, use_tta):
    dataset_path = f"dataset_faceshape_{gender}.npz"
    if not os.path.isfile(dataset_path):
        print(f"[건너뜀] {dataset_path} 없음")
        return None

    data = np.load(dataset_path)
    x_test, y_test = data["x_test"], data["y_test"]

    correct = 0
    for i in range(len(x_test)):
        img_bgr = cv2.cvtColor(x_test[i].astype("uint8"), cv2.COLOR_RGB2BGR)
        pred_shape, _ = cnn_classifier.predict_face_shape(img_bgr, gender, use_tta=use_tta)
        true_shape = cnn_classifier.FACE_SHAPE_CLASSES[int(y_test[i])]
        if pred_shape == true_shape:
            correct += 1

    acc = correct / len(x_test) * 100
    print(f"  [{gender}] use_tta={use_tta}: {correct}/{len(x_test)} = {acc:.1f}%")
    return acc


def main():
    print("=== 가중치 로드 ===")
    cnn_classifier.load_model(".", transfer=True)

    print("\n=== 전체 테스트셋 기준 TTA 전/후 비교 ===")
    print("(테스트셋 크기에 따라 몇 분 걸릴 수 있습니다)\n")

    for gender in ("men", "women"):
        print(f"--- {gender} ---")
        acc_no_tta = evaluate(gender, use_tta=False)
        acc_tta = evaluate(gender, use_tta=True)
        if acc_no_tta is not None and acc_tta is not None:
            diff = acc_tta - acc_no_tta
            sign = "+" if diff >= 0 else ""
            print(f"  차이: {sign}{diff:.1f}%p\n")


if __name__ == "__main__":
    main()
