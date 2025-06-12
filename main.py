import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3
import time
import 

# Inicializações
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()

X = []  # amostras
y = []  # rótulos
modelo_treinado = False
knn = KNeighborsClassifier(n_neighbors=3)

def extrair_vetor(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

cap = cv2.VideoCapture(0)
ultimo_gesto = ""
ultimo_tempo = 0
mensagem = ""

print("[INSTRUÇÕES]")
print("1️⃣ Faça o gesto de 'oi, meu nome é Victor'")
print("2️⃣ Pressione a tecla 'T' várias vezes para registrar exemplos")
print("3️⃣ Pressione 'R' para treinar o modelo")
print("4️⃣ Depois, ele reconhece o gesto automaticamente")
print("5️⃣ Pressione 'ESC' para sair")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        vetor = extrair_vetor(hand_landmarks.landmark)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            X.append(vetor)
            y.append("Oi, meu nome é Victor")
            print(f"[✓] Exemplo {len(X)} coletado.")

        elif key == ord('r'):
            if len(X) >= 3:
                knn.fit(X, y)
                modelo_treinado = True
                print("[✓] Modelo treinado com sucesso!")
            else:
                print("[!] Adicione mais exemplos antes de treinar.")

        if modelo_treinado:
            pred = knn.predict([vetor])[0]
            distancia, _ = knn.kneighbors([vetor], n_neighbors=1, return_distance=True)

            if distancia[0][0] < 0.2:  # Limiar ajustável
                if pred != ultimo_gesto and time.time() - ultimo_tempo > 2:
                    engine.say(pred)
                    engine.runAndWait()
                    ultimo_gesto = pred
                    ultimo_tempo = time.time()

                mensagem = pred
            else:
                mensagem = "Gestos diferentes"

        # Mostrar resultado na tela
        cv2.putText(frame, mensagem, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Mão não detectada", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Reconhecimento de Gesto - KNN", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
