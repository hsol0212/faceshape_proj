import cv2
import mediapipe as mp
import numpy as np
from core.glasses_ar import GlassesAR

class LandmarkAnalyzer:
    def __init__(self, glasses_path="assets/glasses1.png"):
        # MediaPipe Face Mesh 초기화 (안전한 버전 규격)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 안경 AR 객체 생성
        try:
            self.glasses_ar = GlassesAR(glasses_path)
        except Exception as e:
            print(f"경고: {e}")
            self.glasses_ar = None

    def run_webcam_test(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        print("=== 안경 AR + 랜드마크 테스트 시작 (종료하려면 'q' 키를 누르세요) ===")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("프레임을 읽어올 수 없습니다.")
                break

            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # BGR 이미지를 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # 얼굴 랜드마크가 검출된 경우
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. 얼굴 망(Mesh) 그리기 (필요에 따라 주석 처리 가능)
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # 2. 안경 AR 합성 적용
                    if self.glasses_ar:
                        frame = self.glasses_ar.overlay_glasses(frame, face_landmarks.landmark, frame.shape)

            # 화면 출력
            cv2.imshow("Faceshape Project - AR Glasses Test", frame)

            # 'q'를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = LandmarkAnalyzer(glasses_path="assets/glasses1.png")
    analyzer.run_webcam_test()