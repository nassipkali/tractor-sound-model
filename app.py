import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import joblib
import io
import os
from pydub import AudioSegment
from panns_inference import AudioTagging

# ======================
# НАСТРОЙКИ
# ======================
SR = 16000
MAX_SEC = 6.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
LR_PATH = f"{MODEL_DIR}/panns_lr.joblib"
XGB_PATH = f"{MODEL_DIR}/panns_xgb.joblib"
SCALER_PATH = f"{MODEL_DIR}/panns_scaler.joblib"

# ======================
# АУДИО ЗАГРУЗКА
# ======================
def load_any_audio(file_bytes, sr=SR):
    """Загружает аудио (wav, mp3, m4a, flac) → np.float32 [-1, 1]."""
    try:
        y, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
            audio = audio.set_frame_rate(sr).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if samples.dtype.kind in ("i", "u"):
                maxval = np.iinfo(samples.dtype).max
                samples = samples / (maxval + 1e-9)
            samples = samples / (np.max(np.abs(samples)) + 1e-9)
            return samples
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении аудио: {e}")

# ======================
# PANNs ЭМБЕДДИНГИ
# ======================
@st.cache_resource
def load_panns():
    """Загружает предобученную модель PANNs (AudioTagging)."""
    return AudioTagging(checkpoint_path=None, device=DEVICE)

def extract_panns_embedding(model, y):
    """Извлекает эмбеддинг из PANNs (вектор признаков)."""
    with torch.no_grad():
        out = model.inference(torch.tensor(y[None, :]).to(DEVICE))
    if isinstance(out, dict):
        emb = out.get("embedding", out.get("clipwise_output", None))
    elif isinstance(out, (tuple, list)):
        emb = out[1] if len(out) > 1 else out[0]
    else:
        emb = out
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    emb = np.squeeze(np.array(emb))
    return emb

# ======================
# ВИЗУАЛИЗАЦИЯ
# ======================
def plot_spectrogram(y, sr=SR, title="Log-Mel спектрограмма"):
    """Рисует логарифмическую мел-спектрограмму."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(logmel, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax, cmap='viridis')
    ax.set(title=title)
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    st.pyplot(fig)

# ======================
# СТИЛИ CSS
# ======================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .result-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .result-error {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2d3748;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 2rem 0;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .feature-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# ИНТЕРФЕЙС
# ======================
st.set_page_config(
    page_title="Анализ звука трактора | AI Diagnostics",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.markdown("""
<div class="main-header">
    <h1>🚜 Система анализа звука трактора</h1>
    <p style="color: white; margin: 0; font-size: 1.1rem;">Искусственный интеллект для диагностики техники</p>
</div>
""", unsafe_allow_html=True)

# Боковая панель с информацией
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
    st.markdown("## 📊 О системе")
    st.markdown("""
    Эта система использует глубокое обучение для анализа аудиозаписей работы трактора.
    
    **Возможности:**
    - Детекция аномалий в работе двигателя
    - Анализ спектрограммы звука
    - Два алгоритма ML на выбор
    
    **Поддерживаемые форматы:**
    - WAV, MP3, M4A, FLAC
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Технические параметры")
    st.info(f"""
    **Частота дискретизации:** {SR} Hz  
    **Макс. длительность:** {MAX_SEC} сек  
    **Устройство:** {DEVICE.upper()}
    """)

# Информационная карточка
st.markdown("""
<div class="info-card">
    <h3 style="margin-top: 0;">🎯 Как это работает?</h3>
    <p>Загрузите аудиофайл с записью работы трактора. Модель на основе <strong>PANNs (Pretrained Audio Neural Networks)</strong> 
    извлечет акустические признаки и классифицирует звук.</p>
    <div>
        <span class="feature-badge">✅ Исправная работа</span>
        <span class="feature-badge">🚨 Аномалия/Неисправность</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Секция загрузки
st.markdown("### 📁 Загрузка аудиофайла")
uploaded = st.file_uploader(
    "Перетащите файл сюда или нажмите для выбора",
    type=["wav", "mp3", "m4a", "flac"],
    help="Поддерживаются форматы: WAV, MP3, M4A, FLAC"
)

if not uploaded:
    st.info("👆 Загрузите аудиофайл для начала анализа")
    st.stop()

# Обработка файла
with st.spinner('🔄 Загрузка аудиофайла...'):
    file_bytes = uploaded.read()
    try:
        y = load_any_audio(file_bytes, sr=SR)
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")
        st.stop()

# Информация о файле
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📄 Имя файла", uploaded.name)
with col2:
    st.metric("⏱️ Длительность", f"{len(y)/SR:.2f} сек")
with col3:
    st.metric("💾 Размер", f"{len(file_bytes)/1024:.1f} KB")

st.markdown("---")

# Аудио плеер
st.markdown("### 🔊 Прослушивание")
st.audio(uploaded)

# Спектрограмма
st.markdown("### 📈 Визуализация спектрограммы")
with st.spinner('🎨 Построение спектрограммы...'):
    plot_spectrogram(y, sr=SR, title="Log-Mel спектрограмма аудиосигнала")

st.markdown("---")

# ======================
# ЗАГРУЗКА МОДЕЛЕЙ
# ======================
available_models = {}
if os.path.exists(LR_PATH):
    available_models["Logistic Regression (по умолчанию)"] = joblib.load(LR_PATH)
if os.path.exists(XGB_PATH):
    available_models["XGBoost"] = joblib.load(XGB_PATH)

if not available_models:
    st.error("❌ Не найдены модели в папке /models (ожидаются panns_lr.joblib или panns_xgb.joblib).")
    st.stop()

try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ Не найден файл масштабирования: models/panns_scaler.joblib")
    st.stop()

# Выбор модели
st.markdown("### 🤖 Выбор модели машинного обучения")
model_choice = st.radio(
    "Выберите алгоритм классификации:",
    list(available_models.keys()),
    index=0,
    help="Logistic Regression - быстрая и надежная. XGBoost - более сложная, потенциально точнее."
)
clf = available_models[model_choice]

with st.spinner('⚙️ Загрузка модели PANNs...'):
    panns_model = load_panns()

# ======================
# АНАЛИЗ
# ======================
st.markdown("---")
if st.button("🔍 ЗАПУСТИТЬ АНАЛИЗ", use_container_width=True):
    with st.spinner('🧠 Анализ аудио... Это может занять несколько секунд'):
        y_truncated = y[:int(SR * MAX_SEC)]
        emb = extract_panns_embedding(panns_model, y_truncated)
        X = scaler.transform([emb])
        prob = clf.predict_proba(X)[0][1] if hasattr(clf, "predict_proba") else float(clf.predict(X)[0])
        pred = int(prob > 0.5)
    
    st.markdown("---")
    st.markdown("## 📊 Результаты анализа")
    
    # Результат
    if pred == 1:
        st.markdown(f"""
        <div class="result-error">
            <div class="result-title">🚨 ОБНАРУЖЕНА АНОМАЛИЯ</div>
            <p style="font-size: 1.2rem; color: #2d3748;">Система обнаружила признаки неисправности в работе трактора</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-success">
            <div class="result-title">✅ ТРАКТОР ИСПРАВЕН</div>
            <p style="font-size: 1.2rem; color: #2d3748;">Аудиосигнал соответствует нормальной работе</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "🎯 Вероятность аномалии",
            f"{prob:.1%}",
            delta=f"{(prob - 0.5)*100:.1f}% от порога" if prob != 0.5 else "На пороге"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🤖 Модель", model_choice.split()[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚖️ Порог классификации", "50%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Прогресс-бар вероятности
    st.markdown("### 📊 Визуализация уверенности модели")
    st.progress(prob)
    
    # Рекомендации
    st.markdown("---")
    st.markdown("### 💡 Рекомендации")
    if pred == 1:
        st.warning("""
        **Рекомендуется:**
        - Провести визуальный осмотр трактора
        - Проверить уровень масла и других жидкостей
        - Обратиться к специалисту для детальной диагностики
        - Избегать интенсивной эксплуатации до устранения проблемы
        """)
    else:
        st.success("""
        **Результат:**
        - Звук соответствует нормальной работе
        - Признаков явных неисправностей не обнаружено
        - Рекомендуется плановое техническое обслуживание
        """)