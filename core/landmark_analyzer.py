import cv2
import mediapipe as mp

class LandmarkAnalyzer:
    def __init__(self):
        # MediaPipe Face Mesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 검증이 필요한 핵심 랜드마크 인덱스 (초안)
        self.target_indices = {
            "Left Temple": 162,   # 왼쪽 관자놀이
            "Right Temple": 389,  # 오른쪽 관자놀이
            "Chin": 152,          # 턱 끝
            "Left Jaw": 132,      # 왼쪽 귀밑 턱선
            "Right Jaw": 361,     # 오른쪽 귀밑 턱선
            "Mid Between Eyes": 168 # 미간
        }

    def verify_landmarks_live(self):
        """웹캠을 켜고 특정 랜드마크 위치를 화면에 그려서 눈으로 검증하는 디버그 함수"""
        cap = cv2.VideoCapture(0) # 0번 웹캠 켜기

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("웹캠을 찾을 수 없습니다. 카메라 연결을 확인해주세요.")
                break

            # MediaPipe는 RGB 이미지를 사용하므로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, c = image.shape
                    
                    # 우리가 추적하려는 특정 점들만 화면에 크게 그리기
                    for name, index in self.target_indices.items():
                        point = face_landmarks.landmark[index]
                        cx, cy = int(point.x * w), int(point.y * h)
                        
                        # 빨간색 원으로 점 찍기
                        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                        # 점 옆에 이름과 번호 텍스트 띄우기
                        cv2.putText(image, f"{name}({index})", (cx + 10, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 화면 출력
            cv2.imshow('Landmark Verification - Press ESC to exit', image)

            # ESC 키 누르면 종료
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = LandmarkAnalyzer()
    analyzer.verify_landmarks_live()