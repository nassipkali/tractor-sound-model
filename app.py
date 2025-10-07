import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tempfile
import os

# === НАСТРОЙКИ ===
SR = 16000
N_MFCC = 40

# === ЗАГРУЗКА МОДЕЛИ ===
model = joblib.load("tractor_model.pkl")
scaler = joblib.load("tractor_scaler.pkl")

# === ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    y = y / np.max(np.abs(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.median(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])
    return features, mfcc, y, sr

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(page_title="Диагностика трактора по звуку", page_icon="🚜", layout="centered")

st.title("🎧 Диагностика звука трактора")
st.markdown("Загрузите аудиофайл (`.wav` или `.m4a`), затем нажмите **Проанализировать**, чтобы определить наличие аномалий в работе двигателя.")

# === ЗАГРУЗКА ФАЙЛА ===
uploaded_file = st.file_uploader("Загрузите аудиофайл", type=["wav", "m4a"])

if uploaded_file is not None:
    # Временное сохранение файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Отображаем плеер
    st.audio(uploaded_file, format='audio/wav')

    # Кнопка анализа
    analyze_button = st.button("🔍 Проанализировать")

    if analyze_button:
        try:
            with st.spinner("Идёт анализ звука... ⏳"):
                features, mfcc, y, sr = extract_features(temp_path)
                X_scaled = scaler.transform(features.reshape(1, -1))
                prob = model.predict_proba(X_scaled)[0, 1]
                result = "⚠️ АНОМАЛИЯ" if prob > 0.5 else "✅ НОРМА"

                # Вывод результата
                st.subheader(f"Результат: {result}")
                st.metric(label="Вероятность аномалии", value=f"{prob:.2f}")

                # Визуализация MFCC
                st.write("Спектрограмма MFCC:")
                fig, ax = plt.subplots(figsize=(8, 3))
                librosa.display.specshow(mfcc, sr=sr, x_axis="time")
                plt.colorbar(format="%+2.0f dB")
                plt.title("MFCC (мел-частотные коэффициенты)")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Ошибка при анализе файла: {e}")

        # Удаляем временный файл
        os.remove(temp_path)
