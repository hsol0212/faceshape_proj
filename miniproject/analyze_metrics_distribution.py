# -*- coding: utf-8 -*-
"""
analyze_metrics_distribution.py
==================================
men/, women/ 폴더의 사진 전체에 랜드마크 세부 수치 계산을 돌려서,
얼굴형 클래스(Round/Oval/Square/Rectangle)별로 face_width_ratio,
jaw_width_ratio, midface_ratio 값이 어떻게 분포하는지 통계로 뽑습니다.

이 결과로 THRESH_NARROW_TEMPLE 등의 값을 "감"이 아니라 실제 분포 기반으로
정할 수 있습니다 (예: Square 클래스의 jaw_width_ratio 중앙값 근처를 경계로 삼기).

실행 위치: faceshape_project/ 안에서 실행 (men/, women/ 폴더와 같은 위치)
설치 필요: pip install mediapipe pandas --break-system-packages
"""

import glob
import os
import cv2
import pandas as pd

from core import landmark_analyzer as la

CLASS_NAMES = ["Oval", "Rectangle", "Round", "Square"]
SOURCE_ROOTS = {"men": r".\men", "women": r".\women"}

# 사진이 많으면 시간이 오래 걸리니, 클래스당 최대 이 개수만큼만 샘플링
# (None으로 하면 전체 다 돌림 — 처음엔 작게 테스트해보는 걸 추천)
MAX_PER_CLASS = 150


def process_folder(folder_path, gender, face_shape, rows):
    files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )
    if MAX_PER_CLASS:
        files = files[:MAX_PER_CLASS]

    ok, no_face = 0, 0
    for filepath in files:
        img = cv2.imread(filepath)
        if img is None:
            continue
        _, landmarks = la.detect_face_and_landmarks(img)
        if landmarks is None:
            no_face += 1
            continue
        metrics = la.extract_face_metrics(landmarks)
        if metrics is None:
            no_face += 1
            continue
        rows.append({
            "gender": gender,
            "face_shape": face_shape,
            "file": os.path.basename(filepath),
            **metrics,
        })
        ok += 1

    print(f"  [{gender}/{face_shape}] 처리 {ok}장 성공, {no_face}장 얼굴 검출 실패 (총 {len(files)}장 중)")


def main():
    rows = []
    for gender, root in SOURCE_ROOTS.items():
        if not os.path.isdir(root):
            print(f"경고: {root} 없음, 건너뜀")
            continue
        print(f"\n=== {gender} ===")
        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            matched = next((c for c in CLASS_NAMES if c.lower() == entry.lower()), None)
            if matched is None:
                continue
            process_folder(entry_path, gender, matched, rows)

    df = pd.DataFrame(rows)
    df.to_csv("metrics_distribution_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n원본 데이터 저장: metrics_distribution_raw.csv ({len(df)}행)")

    print("\n" + "=" * 70)
    print(" 클래스별 통계 (평균 / 중앙값 / 25%~75% 구간)")
    print("=" * 70)

    for metric in ["face_width_ratio", "jaw_width_ratio", "face_height_ratio", "midface_ratio"]:
        print(f"\n--- {metric} ---")
        summary = df.groupby(["gender", "face_shape"])[metric].agg(
            ["mean", "median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
        )
        summary.columns = ["mean", "median", "q25", "q75"]
        print(summary.round(3))

    print("\n※ 이 통계를 참고해서 core/landmark_analyzer.py의 THRESH_* 값을 정하세요.")
    print("  예: jaw_width_ratio가 Square에서 유독 높게 나오면, Square의 q25~median 사이")
    print("  값을 THRESH_WIDE_JAW로 잡는 식으로 접근하면 근거가 생깁니다.")


if __name__ == "__main__":
    main()
