import cv2
import mediapipe as mp
import joblib
import numpy as np

# Carregar o modelo treinado e o label encoder
model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Captura de vídeo
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para RGB e detectar mãos
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_name = "Nenhum gesto"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coletar pontos da mão
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Converter para array NumPy e fazer a predição
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)
            gesture_name = label_encoder.inverse_transform(prediction)[0]

    # Mostrar nome do gesto na tela
    cv2.putText(frame, f"Gesto: {gesture_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Reconhecimento de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
