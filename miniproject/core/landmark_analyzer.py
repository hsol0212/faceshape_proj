# -*- coding: utf-8 -*-
"""
core/landmark_analyzer.py   [갈래 B 담당]
==========================================
얼굴 검출 + 랜드마크 좌표 + 세부 수치(관자놀이/턱/중안부) + 수식어 태그를 담당합니다.
CNN 쪽(core/cnn_classifier.py)과는 완전히 독립적인 파일입니다.

mediapipe가 설치돼 있어야 합니다: pip install mediapipe --break-system-packages

다른 파일에서 쓰는 인터페이스:
    detect_face_and_landmarks(frame_bgr) -> (face_crop, landmarks dict) | (None, None)
    extract_face_metrics(landmarks) -> dict | None
    get_modifier_tags(metrics) -> list[str]
이 함수 시그니처만 유지하면 내부 구현(인덱스, 임계값 등)은 자유롭게 바꿔도 됩니다.

※ 관자놀이/턱선/미간 인덱스는 팀원이 웹캠 디버그 도구(landmark_analyzer.py 검증 스크립트)로
   확인 완료했습니다 (관자놀이/턱선/미간/이마상단/코끝/코밑 6개 전부).
   임계값(THRESH_*)은 아직 실측 조정 전입니다.
"""

import mediapipe as mp

_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# FaceMesh 랜드마크 인덱스 (양쪽 눈 중심 근사치)
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

# 세부 수치 측정에서 쓰는 추가 포인트
# 참고: mediapipe Face Mesh 468포인트 토폴로지 기준 근사 인덱스 — 검증 필요
# 팀원이 웹캠 디버그 검증을 마친 인덱스 (landmark_analyzer.py 검증 도구로 확인함)
FOREHEAD_TOP_IDX = 10        # 이마 최상단 (헤어라인 근처) — 검증 완료
CHIN_BOTTOM_IDX = 152        # 턱 최하단 — 검증 완료
LEFT_TEMPLE_IDX = 162        # 왼쪽 관자놀이 — 검증 완료
RIGHT_TEMPLE_IDX = 389       # 오른쪽 관자놀이 — 검증 완료
LEFT_JAW_IDX = 132           # 왼쪽 귀밑 턱선 — 검증 완료
RIGHT_JAW_IDX = 361          # 오른쪽 귀밑 턱선 — 검증 완료
NOSE_TIP_IDX = 1             # 코끝 — 검증 완료
BROW_MID_IDX = 168           # 미간 — 검증 완료
NOSE_BOTTOM_IDX = 2          # 코 밑 — 검증 완료


def detect_face_and_landmarks(frame_bgr):
    """얼굴을 찾아 (crop된 얼굴 이미지, 랜드마크 dict) 반환. 못 찾으면 (None, None)."""
    import cv2  # 이 함수에서만 필요 — cnn_classifier.py와 임포트 의존 분리 목적

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0].landmark

    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1, x2 = max(min(xs), 0), min(max(xs), w)
    y1, y2 = max(min(ys), 0), min(max(ys), h)

    # 정사각형에 가깝게 살짝 여유를 두고 crop (배경/머리카락 과적합 방지 목적)
    pad = int(0.15 * (x2 - x1))
    x1, x2 = max(x1 - pad, 0), min(x2 + pad, w)
    y1, y2 = max(y1 - pad, 0), min(y2 + pad, h)

    face_crop = frame_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    landmarks = {
        "left_eye": (int(lm[LEFT_EYE_IDX].x * w), int(lm[LEFT_EYE_IDX].y * h)),
        "right_eye": (int(lm[RIGHT_EYE_IDX].x * w), int(lm[RIGHT_EYE_IDX].y * h)),
        "forehead_top": (int(lm[FOREHEAD_TOP_IDX].x * w), int(lm[FOREHEAD_TOP_IDX].y * h)),
        "chin_bottom": (int(lm[CHIN_BOTTOM_IDX].x * w), int(lm[CHIN_BOTTOM_IDX].y * h)),
        "left_temple": (int(lm[LEFT_TEMPLE_IDX].x * w), int(lm[LEFT_TEMPLE_IDX].y * h)),
        "right_temple": (int(lm[RIGHT_TEMPLE_IDX].x * w), int(lm[RIGHT_TEMPLE_IDX].y * h)),
        "left_jaw": (int(lm[LEFT_JAW_IDX].x * w), int(lm[LEFT_JAW_IDX].y * h)),
        "right_jaw": (int(lm[RIGHT_JAW_IDX].x * w), int(lm[RIGHT_JAW_IDX].y * h)),
        "nose_tip": (int(lm[NOSE_TIP_IDX].x * w), int(lm[NOSE_TIP_IDX].y * h)),
        "brow_mid": (int(lm[BROW_MID_IDX].x * w), int(lm[BROW_MID_IDX].y * h)),
        "nose_bottom": (int(lm[NOSE_BOTTOM_IDX].x * w), int(lm[NOSE_BOTTOM_IDX].y * h)),
    }
    return face_crop, landmarks


# ------------------------------------------------------------------
# 세부 수치 측정 + 수식어 태그
# ------------------------------------------------------------------

def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def extract_face_metrics(landmarks):
    """랜드마크 dict -> 세부 수치 dict.
    얼굴 크기(눈 사이 거리)에 대한 비율로 정규화해서, 카메라 거리 차이에 영향을 덜 받게 합니다.
    """
    eye_dist = _dist(landmarks["left_eye"], landmarks["right_eye"])
    if eye_dist <= 0:
        return None

    face_width = _dist(landmarks["left_temple"], landmarks["right_temple"])
    jaw_width = _dist(landmarks["left_jaw"], landmarks["right_jaw"])
    face_height = _dist(landmarks["forehead_top"], landmarks["chin_bottom"])

    # 중안부 비율: (미간~코밑) / (코밑~턱끝) — 값이 클수록 "중안부가 길다"
    midface = _dist(landmarks["brow_mid"], landmarks["nose_bottom"])
    lowerface = _dist(landmarks["nose_bottom"], landmarks["chin_bottom"])
    midface_ratio = midface / lowerface if lowerface > 0 else None

    return {
        "face_width_ratio": face_width / eye_dist,     # 관자놀이 폭 (눈 사이 거리 대비)
        "jaw_width_ratio": jaw_width / eye_dist,        # 하악각 폭
        "face_height_ratio": face_height / eye_dist,    # 세로 길이
        "midface_ratio": midface_ratio,                 # 중안부 길이감
    }


# 임계값 — 실측 데이터로 재조정 필요 (초기 추정값)
# 임계값 — 2026-07-27 실측 데이터(men+women 사진 1,001장) 기준으로 재조정.
# 기존 값(3.6, 3.4, 1.15)은 스케일 자체가 틀려서 narrow_temple이 항상 참,
# wide_jaw/long_midface가 항상 거짓으로 나오는 심각한 버그였음 — 이번에 수정.
# 전체(남녀 통합) 중앙값 근처로 잡음: face_width_ratio 중앙값 약 1.55,
# jaw_width_ratio 약 1.49, midface_ratio 약 0.77.
# ※ 성별에 따라 실측값이 체계적으로 다르게 나와서(여성이 전반적으로 낮음),
#   더 정교하게 하려면 성별별로 다른 임계값을 쓰는 게 이상적임 — 지금은 우선
#   전체 통합 기준으로 스케일 버그부터 고친 상태.
THRESH_NARROW_TEMPLE = 1.55      # 이보다 작으면 "관자놀이가 좁다"
THRESH_WIDE_JAW = 1.49           # 이보다 크면 "턱선이 각지고 넓다"
THRESH_LONG_MIDFACE = 0.77       # 이보다 크면 "중안부가 길다"


def get_modifier_tags(metrics):
    """세부 수치 -> 모듈형 추천에 쓰일 태그 리스트 (예: ['long_midface', 'wide_jaw'])"""
    if metrics is None:
        return []
    tags = []
    if metrics["face_width_ratio"] < THRESH_NARROW_TEMPLE:
        tags.append("narrow_temple")       # 관자놀이 여백 있음 → 옆볼륨/사이드뱅으로 보완
    if metrics["jaw_width_ratio"] > THRESH_WIDE_JAW:
        tags.append("wide_jaw")            # 하악각 각짐 → 다운펌/턱선 가리는 스타일
    if metrics["midface_ratio"] and metrics["midface_ratio"] > THRESH_LONG_MIDFACE:
        tags.append("long_midface")        # 중안부 길음 → 풀뱅/시스루뱅으로 보완
    return tags


# ------------------------------------------------------------------
# 디버그: 랜드마크 포인트를 웹캠 화면에 그려서 눈으로 검증하는 도구
# (팀원이 만든 LandmarkAnalyzer.verify_landmarks_live()를 함수 형태로 옮김 —
#  클래스 대신 이 파일의 다른 함수들과 같은 스타일로 유지)
# 단독 실행: python core/landmark_analyzer.py
# ------------------------------------------------------------------

_DEBUG_TARGET_INDICES = {
    "Left Temple": LEFT_TEMPLE_IDX,
    "Right Temple": RIGHT_TEMPLE_IDX,
    "Chin": CHIN_BOTTOM_IDX,
    "Left Jaw": LEFT_JAW_IDX,
    "Right Jaw": RIGHT_JAW_IDX,
    "Mid Between Eyes": BROW_MID_IDX,
    "Forehead Top": FOREHEAD_TOP_IDX,   # 검증 완료
    "Nose Tip": NOSE_TIP_IDX,           # 검증 완료
    "Nose Bottom": NOSE_BOTTOM_IDX,     # 검증 완료
}


def debug_visualize_landmarks():
    """웹캠을 켜고 랜드마크 포인트를 화면에 그려서 눈으로 검증하는 디버그 함수.
    ESC로 종료. 이 파일을 단독 실행하면 바로 이 함수가 돌아갑니다."""
    import cv2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("웹캠을 찾을 수 없습니다. 카메라 연결을 확인해주세요.")
        return

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _mp_face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = image.shape[:2]
                for name, index in _DEBUG_TARGET_INDICES.items():
                    point = face_landmarks.landmark[index]
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(image, f"{name}({index})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Landmark Verification - Press ESC to exit", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_visualize_landmarks()
