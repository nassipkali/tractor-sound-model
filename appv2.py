import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
import joblib
import subprocess
import warnings

# =======================
# НАСТРОЙКИ
# =======================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "tractor_crnn_v2.h5"
CONFIG_PATH = "tractor_config.pkl"

# Загружаем модель и параметры
model = load_model(MODEL_PATH)
config = joblib.load(CONFIG_PATH)
SR = config["sr"]
N_MELS = config["n_mels"]
MAX_LEN = config["max_len"]

# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================
def convert_to_wav(uploaded_file):
    """Преобразует m4a/mp3/wav → временный wav через ffmpeg"""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in_path = tmp_in.name

    tmp_out_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ac", "1", "-ar", str(SR), tmp_out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return tmp_out_path
    except Exception as e:
        st.error(f"Не удалось конвертировать аудио: {e}")
        return None
    finally:
        os.remove(tmp_in_path)


def extract_logmel(file_path):
    """Извлекает Log-Mel спектрограмму"""
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    y = y / np.max(np.abs(y))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Паддинг / обрезка
    if logmel.shape[1] < MAX_LEN:
        pad = MAX_LEN - logmel.shape[1]
        logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="constant")
    else:
        logmel = logmel[:, :MAX_LEN]
    return logmel[..., np.newaxis]


def predict_anomaly(file_path):
    """Вычисляет вероятность аномалии"""
    logmel = extract_logmel(file_path)
    X = np.expand_dims(logmel, axis=0)
    prob = model.predict(X, verbose=0)[0, 0]
    return prob, logmel

# =======================
# STREAMLIT ИНТЕРФЕЙС
# =======================
st.set_page_config(page_title="Анализ звука трактора", page_icon="🚜", layout="centered")

st.title("🚜 Анализ звука трактора")
st.markdown("Определение технического состояния по аудиозаписи двигателя (Log-Mel + CRNN).")

uploaded_file = st.file_uploader("Загрузите аудио", type=["wav", "m4a", "mp3"])

if uploaded_file:
    temp_wav = convert_to_wav(uploaded_file)
    if temp_wav:
        st.audio(temp_wav, format="audio/wav")

        analyze = st.button("🔍 Выполнить анализ")

        if analyze:
            with st.spinner("Анализируем звук..."):
                try:
                    prob, logmel = predict_anomaly(temp_wav)
                    st.success("✅ Анализ завершён!")

                    # === Визуальный индикатор ===
                    st.subheader("🧠 Результат анализа")
                    if prob > 0.5:
                        st.markdown(
                            f"<h2 style='color:red;'>🔴 Обнаружена аномалия ({prob*100:.1f}%)</h2>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<h2 style='color:green;'>🟢 Трактор работает исправно ({(1-prob)*100:.1f}%)</h2>",
                            unsafe_allow_html=True
                        )

                    # === График Log-Mel ===
                    st.subheader("🎧 Log-Mel спектрограмма")
                    fig, ax = plt.subplots(figsize=(8, 3))
                    img = librosa.display.specshow(
                        logmel.squeeze(),
                        sr=SR,
                        hop_length=512,
                        x_axis="time",
                        y_axis="mel",
                        cmap="magma",
                        ax=ax
                    )
                    ax.set(title="Log-Mel спектрограмма (дБ)")
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ошибка при анализе: {type(e).__name__}")
                    st.code(str(e))

        os.remove(temp_wav)
