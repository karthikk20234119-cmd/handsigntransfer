# train_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import sys

# Resolve project root and paths robustly (works when run from repo root or from src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(BASE_DIR, "datasets", "data_all.csv")
MODEL_OUT = os.path.join(BASE_DIR, "models", "gesture_model.h5")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# Sanity check: helpful error if dataset is missing
if not os.path.exists(DATA_CSV):
    datasets_dir = os.path.join(BASE_DIR, "datasets")
    available = []
    if os.path.isdir(datasets_dir):
        try:
            available = os.listdir(datasets_dir)
        except Exception:
            available = ["<unreadable directory>"]
    raise FileNotFoundError(
        f"Dataset not found at {DATA_CSV}.\n" \
        f"Looking for file 'data_all.csv' inside '{datasets_dir}'.\n" \
        f"datasets/ contents: {available}"
    )

df = pd.read_csv(DATA_CSV)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_enc, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)
model.save(MODEL_OUT)
# Save label encoder mapping
import pickle
with open("../models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Model & label encoder saved.")
