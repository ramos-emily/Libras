import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Inicializando MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Criar ou carregar dataset
DATASET_FILE = "gestures.csv"

if os.path.exists(DATASET_FILE):
    dataset = pd.read_csv(DATASET_FILE)
else:
    dataset = pd.DataFrame(columns=["gesture"] + [f"x{i},y{i},z{i}" for i in range(63)])

# Captura de vídeo
cap = cv2.VideoCapture(0)

gesture_name = input("Digite o nome do gesto que deseja capturar: ")
num_samples = int(input("Quantas amostras deseja capturar? "))  # Define quantas capturas serão feitas
print(f"Capturando {num_samples} amostras do gesto '{gesture_name}', uma a cada 5 segundos...")

start_time = time.time()
sample_count = 0

while cap.isOpened() and sample_count < num_samples:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Se passaram 5 segundos desde a última captura?
            if time.time() - start_time >= 5:
                # Coletar coordenadas dos pontos da mão
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Adicionar nova amostra ao dataset
                new_data = pd.DataFrame([[gesture_name] + landmarks], columns=dataset.columns)
                dataset = pd.concat([dataset, new_data], ignore_index=True)
                sample_count += 1
                print(f"Amostra {sample_count}/{num_samples} do gesto '{gesture_name}' salva!")

                start_time = time.time()  # Reinicia o tempo

    cv2.imshow("Captura de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Salvar dataset
dataset.to_csv(DATASET_FILE, index=False)
print(f"Dataset salvo como {DATASET_FILE}")

cap.release()
cv2.destroyAllWindows()
