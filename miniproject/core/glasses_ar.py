import cv2
import numpy as np

class GlassesAR:
    def __init__(self, glasses_path):
        # 알파 채널(투명도, PNG)을 포함하여 안경 이미지 읽기 (-1 옵션)
        self.glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
        
        if self.glasses_img is None:
            raise FileNotFoundError(f"안경 이미지 경로를 찾을 수 없습니다: {glasses_path}")

    def overlay_glasses(self, frame, landmarks, image_shape):
        """
        랜드마크 좌표를 기반으로 얼굴에 안경을 합성하는 함수
        - landmarks: MediaPipe로 검출된 랜드마크 리스트 (픽셀 좌표계)
        - image_shape: 웹캠 프레임의 (height, width)
        """
        h, w, _ = image_shape

        try:
            # MediaPipe Face Mesh 기준 대표적인 눈 주변/미간 좌표 활용
            # 왼쪽 눈 외곽: 33번, 오른쪽 눈 외곽: 263번 (또는 눈썹/코 근처 활용 가능)
            # 여기서는 양쪽 눈 끝을 기준으로 안경의 너비와 중심을 계산합니다.
            left_eye_x = int(landmarks[33].x * w)
            left_eye_y = int(landmarks[33].y * h)
            
            right_eye_x = int(landmarks[263].x * w)
            right_eye_y = int(landmarks[263].y * h)

            # 양쪽 눈의 중심 좌표 계산
            center_x = (left_eye_x + right_eye_x) // 2
            center_y = (left_eye_y + right_eye_y) // 2

            # 눈 사이의 거리를 측정하여 안경 크기(너비) 조절 (안경이 눈보다 넓어야 하므로 배율 곱하기)
            eye_distance = np.linalg.norm(np.array([left_eye_x, left_eye_y]) - np.array([right_eye_x, right_eye_y]))
            glasses_width = int(eye_distance * 2.5) # 안경 크기 배율 조정 값
            
            # 비율에 맞춰 안경 높이 자동 계산
            g_h, g_w, _ = self.glasses_img.shape
            glasses_height = int(glasses_width * (g_h / g_w))

            # 크기 조절된 안경 이미지 생성
            resized_glasses = cv2.resize(self.glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

            # 합성할 위치(Top-Left 기준) 계산 (안경 중심이 눈 중심보다 살짝 위로 오도록 오프셋 조정)
            x1 = center_x - glasses_width // 2
            y1 = center_y - glasses_height // 2 - int(glasses_height * 0.1)

            # 프레임 범위를 벗어나지 않도록 클리핑 처리
            if x1 < 0 or y1 < 0 or x1 + glasses_width > w or y1 + glasses_height > h:
                return frame # 화면 밖으로 나가면 합성 생략

            # 안경 이미지의 알파(투명도) 채널 분리
            if resized_glasses.shape[2] == 4:
                b, g, r, a = cv2.split(resized_glasses)
                mask = cv2.merge((a, a, a)) / 255.0
                glasses_rgb = cv2.merge((b, g, r))

                # ROI (Region of Interest - 안경이 얹어질 웹캠 영역)
                roi = frame[y1:y1+glasses_height, x1:x1+glasses_width].astype(float)

                # 알파 블렌딩 공식을 이용한 자연스러운 합성
                masked_glasses = glasses_rgb.astype(float)
                masked_roi = roi * (1 - mask)
                result_face = (masked_glasses * mask) + masked_roi

                # 원본 프레임에 합성된 영역 덮어쓰기
                frame[y1:y1+glasses_height, x1:x1+glasses_width] = result_face.astype(np.uint8)

        except Exception as e:
            # 좌표 계산 중 예외 발생 시 원본 프레임 유지
            pass

        return frame