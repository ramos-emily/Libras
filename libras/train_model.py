import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar o dataset
DATASET_FILE = "gestures.csv"
df = pd.read_csv(DATASET_FILE)

# Separar labels e features
X = df.iloc[:, 1:].values  # Posições das mãos
y = df.iloc[:, 0].values   # Nome do gesto

# Converter labels para números
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Dividir em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar um modelo RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar o modelo
accuracy = model.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Salvar o modelo e o label encoder
joblib.dump(model, "gesture_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("Modelo salvo como 'gesture_model.pkl'")
