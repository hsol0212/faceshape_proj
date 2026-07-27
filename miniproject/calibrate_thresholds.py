# -*- coding: utf-8 -*-
"""
calibrate_thresholds.py
=========================
core/landmark_analyzer.py의 THRESH_* 값을 실측으로 정하기 위한 도구.
웹캠에 얼굴을 비추면 화면에 실제 계산된 세부 수치(관자놀이 폭 비율, 턱 폭 비율,
중안부 비율)를 숫자로 실시간 표시합니다.

쓰는 법:
  1. 여러 사람(또는 본인이 고개 각도를 바꿔가며) 얼굴을 비춰서 숫자를 관찰
  2. "이 사람은 관자놀이가 좁아 보이는데 face_width_ratio가 얼마로 나오네" 하는 식으로
     실제 인상과 숫자를 짝지어서 기록 (종이에 적어도 되고, 's' 키로 스냅샷 기록해도 됨)
  3. 몇 명 모으면 "대략 이 값 아래면 좁다고 봐도 되겠다" 하는 경계선이 보임
  4. core/landmark_analyzer.py의 THRESH_NARROW_TEMPLE 등을 그 값으로 수정

키 조작: s = 현재 수치를 콘솔에 기록(이름 입력), ESC = 종료
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from core import landmark_analyzer as la

LOG_FILE = "threshold_calibration_log.csv"


def draw_text(frame, text, pos, color=(0, 255, 0), scale=0.6):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("label,face_width_ratio,jaw_width_ratio,face_height_ratio,midface_ratio\n")

    print("s = 현재 수치 기록 / ESC = 종료")
    last_metrics = None

    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        face_crop, landmarks = la.detect_face_and_landmarks(frame)

        if landmarks is not None:
            metrics = la.extract_face_metrics(landmarks)
            last_metrics = metrics
            if metrics:
                y = 30
                for key, val in metrics.items():
                    text = f"{key}: {val:.3f}" if val is not None else f"{key}: N/A"
                    draw_text(frame, text, (10, y))
                    y += 30

                tags = la.get_modifier_tags(metrics)
                draw_text(frame, f"tags: {tags}", (10, y), color=(0, 255, 255))

        else:
            draw_text(frame, "얼굴을 찾는 중...", (10, 30), color=(0, 0, 255))

        cv2.imshow("Threshold Calibration - s:기록 ESC:종료", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s') and last_metrics:
            label = input("이 사람/각도에 대한 메모(예: '본인_정면', '친구1_관자좁음'): ")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{label},{last_metrics['face_width_ratio']:.4f},"
                        f"{last_metrics['jaw_width_ratio']:.4f},"
                        f"{last_metrics['face_height_ratio']:.4f},"
                        f"{last_metrics['midface_ratio']:.4f}\n")
            print(f"기록됨: {label}")

    camera.release()
    cv2.destroyAllWindows()
    print(f"\n기록된 데이터: {LOG_FILE} 파일 확인하세요.")


if __name__ == "__main__":
    main()
