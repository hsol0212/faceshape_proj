import sys
import os
import math
import cv2
import mediapipe as mp
import numpy as np

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

class LandmarkAnalyzer:
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def get_distance(self, p1, p2, w, h):
        """두 랜드마크 사이의 2D 픽셀 거리를 계산하는 함수"""
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.hypot(x2 - x1, y2 - y1)

    def get_angle(self, p1, p2, p3, w, h):
        """세 점(p1-p2-p3)이 이루는 각도(p2가 중심)를 계산하는 함수"""
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        x3, y3 = int(p3.x * w), int(p3.y * h)
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = abs(angle)
        return 360 - angle if angle > 180 else angle

    def analyze_face_proportions(self, landmarks, w, h):
        """랜드마크 수치를 기반으로 비율을 계산하고 태그를 반환합니다."""
        tags = []
        metrics = {}

        # ---------------------------------------------------------
        # 주요 랜드마크 인덱스 (MediaPipe Face Mesh 기준)
        # ---------------------------------------------------------
        TOP_HEAD = landmarks[10]    # 이마 끝 (헤어라인 근처)
        CHIN = landmarks[152]       # 턱 끝
        LEFT_CHEEK = landmarks[234] # 왼쪽 광대 외곽
        RIGHT_CHEEK = landmarks[454]# 오른쪽 광대 외곽
        GLABELLA = landmarks[9]     # 미간
        NOSE_BASE = landmarks[94]   # 코 밑 (인중 시작점)
        LIP_TOP = landmarks[13]     # 윗입술 위
        LIP_BOTTOM = landmarks[14]  # 아랫입술 아래
        L_EYE_OUT = landmarks[33]   # 왼쪽 눈꼬리
        L_EYE_IN = landmarks[133]   # 왼쪽 눈앞머리
        L_JAW = landmarks[132]      # 왼쪽 하악각 (귀 밑 턱)
        R_JAW = landmarks[361]      # 오른쪽 하악각

        # ---------------------------------------------------------
        # 1. 안면 가로/세로 비율 (얼굴형 기본 뼈대)
        # ---------------------------------------------------------
        face_height = self.get_distance(GLABELLA, CHIN, w, h) # 미간~턱끝 (헤어라인 편차 제외)
        face_width = self.get_distance(LEFT_CHEEK, RIGHT_CHEEK, w, h)
        hw_ratio = face_height / face_width if face_width > 0 else 0
        metrics['가로세로 비율'] = hw_ratio

        if hw_ratio >= 1.5:
            tags.append("긴 얼굴형")
        elif hw_ratio <= 1.2:
            tags.append("짧은/둥근 얼굴형")

        # ---------------------------------------------------------
        # 2. 상/중/하안부 3등분 비율
        # ---------------------------------------------------------
        upper_face = self.get_distance(TOP_HEAD, GLABELLA, w, h)
        mid_face = self.get_distance(GLABELLA, NOSE_BASE, w, h)
        lower_face = self.get_distance(NOSE_BASE, CHIN, w, h)
        
        metrics['중안부 대비 하안부 비율'] = lower_face / mid_face if mid_face > 0 else 0
        if metrics['중안부 대비 하안부 비율'] >= 1.15:
            tags.append("하안부가 긴 편 (하관 부각)")
        elif mid_face / lower_face >= 1.15 if lower_face > 0 else 0:
            tags.append("중안부가 긴 편 (성숙한 인상)")

        # ---------------------------------------------------------
        # 3. 관자놀이 여백 (눈 가로길이 vs 눈꼬리~얼굴외곽)
        # ---------------------------------------------------------
        eye_width = self.get_distance(L_EYE_OUT, L_EYE_IN, w, h)
        temple_margin = self.get_distance(L_EYE_OUT, LEFT_CHEEK, w, h)
        temple_ratio = temple_margin / eye_width if eye_width > 0 else 0
        metrics['관자놀이 여백 비율'] = temple_ratio

        if temple_ratio >= 1.5:
            tags.append("넓은 관자놀이 여백 (사이드뱅 추천)")

        # ---------------------------------------------------------
        # 4. 인중 길이 비율
        # ---------------------------------------------------------
        philtrum = self.get_distance(NOSE_BASE, LIP_TOP, w, h)
        lower_chin = self.get_distance(LIP_BOTTOM, CHIN, w, h)
        philtrum_ratio = philtrum / lower_chin if lower_chin > 0 else 0
        metrics['인중 길이 비율'] = philtrum_ratio

        if philtrum_ratio >= 0.66: # 1:1.5 비율 이상일 때
            tags.append("긴 인중")

        # ---------------------------------------------------------
        # 5. 턱선 각도 (하악각 - V라인 vs 사각턱)
        # ---------------------------------------------------------
        # 왼쪽 광대 - 하악각 - 턱끝이 이루는 3D 윤곽 각도 측정
        jaw_angle = self.get_angle(LEFT_CHEEK, L_JAW, CHIN, w, h)
        metrics['턱선 각도'] = jaw_angle

        if jaw_angle <= 115:
            tags.append("각진 턱 (사각턱 특징)")
        elif jaw_angle >= 135:
            tags.append("뾰족한 턱 (V라인)")

        # ---------------------------------------------------------
        # 6. 하관 너비 (광대 폭 대비 턱 폭)
        # ---------------------------------------------------------
        jaw_width = self.get_distance(L_JAW, R_JAW, w, h)
        jaw_width_ratio = jaw_width / face_width if face_width > 0 else 0
        metrics['하관 너비 비율'] = jaw_width_ratio

        if jaw_width_ratio >= 0.8:
            tags.append("넓은 하관 (안정적인 형태)")
        elif jaw_width_ratio <= 0.65:
            tags.append("좁은 하관")

        return metrics, tags

    def run_webcam_test(self):
        cap = cv2.VideoCapture(0)
        
        # 분석 텍스트가 너무 빨리 바뀌지 않도록 프레임 지연
        frame_count = 0 

        print("=== 얼굴 비율 및 특징 분석 시작 ===")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 얼굴 매시 그리기
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )

                    # 10프레임마다 콘솔에 분석 결과 출력 (터미널에서 확인)
                    if frame_count % 10 == 0:
                        metrics, tags = self.analyze_face_proportions(face_landmarks.landmark, w, h)
                        
                        # 화면을 지우고 새로 출력 (터미널 깔끔하게 유지)
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print("====================================")
                        print("📊 실시간 얼굴 수치 분석 결과")
                        print("====================================")
                        for key, value in metrics.items():
                            if "각도" in key:
                                print(f"- {key}: {value:.1f}도")
                            else:
                                print(f"- {key}: {value:.2f}")
                        
                        print("\n🏷️ 생성된 특징 태그:")
                        if not tags:
                            print("  (이상적인 표준 밸런스에 가깝습니다)")
                        for tag in tags:
                            print(f"  > {tag}")
                        print("====================================")

            frame_count += 1
            cv2.imshow("Faceshape Project - Landmark Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = LandmarkAnalyzer()
    analyzer.run_webcam_test()