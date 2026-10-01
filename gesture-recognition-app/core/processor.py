import cv2
import mediapipe as mp
import numpy as np
from .models import get_mediapipe_options, load_custom_models

# Conexões padrão entre os 21 pontos da mão no MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),           # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),      # Dedo Médio
    (9, 13), (13, 14), (14, 15), (15, 16),    # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),   # Dedo Mínimo
    (0, 17)                                   # Base da palma
]

class GestureProcessor:
    def __init__(self):
        self.clf, self.label_encoder = load_custom_models()
        self.options = get_mediapipe_options()
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)

    def _draw_hand_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        
        # 1. Desenha as linhas de conexão (verde)
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            pt1 = (int(p1.x * w), int(p1.y * h))
            pt2 = (int(p2.x * w), int(p2.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # 2. Desenha os pontos articulares (vermelho com contorno branco)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    def process_frame(self, frame, draw_landmarks=True):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        recognition_result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        labels = []
        gesture_image = None
        
        # Inicializa as listas de 63 pontos para as duas mãos (126 features no total)
        left_coords = [0.0] * 63
        right_coords = [0.0] * 63
        hands_detected = []

        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                if draw_landmarks:
                    self._draw_hand_landmarks(frame, hand_landmarks)

                handedness = recognition_result.handedness[i][0].category_name
                hands_detected.append(handedness)
                
                # Normalização idêntica ao que foi feito no treinamento
                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z
                
                norm_coords = []
                for lm in hand_landmarks:
                    norm_coords.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    
                if handedness == "Left":
                    left_coords = norm_coords
                elif handedness == "Right":
                    right_coords = norm_coords
            
            # Concatena as coordenadas exatas como no treinamento
            features = np.array(left_coords + right_coords).reshape(1, -1)
            
            # Classificação
            prediction_idx = self.clf.predict(features)[0]
            prediction_prob = np.max(self.clf.predict_proba(features))
            gesture_name = str(self.label_encoder.inverse_transform([int(prediction_idx)])[0])

            labels.append({
                "hand": " + ".join(hands_detected),
                "gesture": gesture_name,
                "probability": float(prediction_prob)
            })
            
            # Mostra a imagem correspondente se a probabilidade for alta
            if float(prediction_prob) > 0.60:
                gesture_image = f"{gesture_name}.png"

        return frame, labels, gesture_image

    def close(self):
        if self.recognizer:
            self.recognizer.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()