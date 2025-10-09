import os
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
import joblib
import scipy.signal as sps

# ======================
# НАСТРОЙКИ
# ======================
SR = 16000
N_MELS = 128
N_SPLITS = 5
AUGMENT_PROB = 0.3
DATA_DIR = "dataset"
MAX_LEN = 128  # по времени (паддинг/обрезка)
HP_CUTOFF = 50.0

# ======================
# ФИЛЬТР + АУГМЕНТАЦИИ
# ======================
def highpass(y, sr=SR, cutoff=HP_CUTOFF):
    """Удаляет низкочастотные шумы (двигатель, микрофонный гул)."""
    b, a = sps.butter(2, cutoff / (sr / 2), btype='highpass')
    return sps.filtfilt(b, a, y)

def normalize(y):
    return y / (np.max(np.abs(y)) + 1e-9)

def add_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y))
    return np.clip(y + noise_factor * noise, -1.0, 1.0)

def time_shift(y, shift_max=0.2):
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def stretch_audio(y, rate_range=(0.9, 1.1)):
    rate = random.uniform(*rate_range)
    return librosa.resample(y, orig_sr=SR, target_sr=int(SR * rate))

def spec_augment(mel_spec, num_mask=2, freq_mask=10, time_mask=10):
    """SpecAugment — случайное обнуление областей спектра."""
    spec = mel_spec.copy()
    for _ in range(num_mask):
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, spec.shape[0] - f)
        spec[f0:f0+f, :] = 0
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, spec.shape[1] - t)
        spec[:, t0:t0+t] = 0
    return spec

# ======================
# ИЗВЛЕЧЕНИЕ LOG-MEL
# ======================
def extract_logmel(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    y = highpass(y, sr)
    y = normalize(y)

    # --- Аугментации ---
    if augment:
        if random.random() < AUGMENT_PROB: y = add_noise(y)
        if random.random() < AUGMENT_PROB: y = time_shift(y)
        if random.random() < AUGMENT_PROB: y = pitch_shift(y, sr)
        if random.random() < AUGMENT_PROB: y = stretch_audio(y)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=256)
    logmel = librosa.power_to_db(mel, ref=np.max)

    if augment and random.random() < 0.3:
        logmel = spec_augment(logmel)

    # Паддинг / обрезка
    if logmel.shape[1] < MAX_LEN:
        pad = MAX_LEN - logmel.shape[1]
        logmel = np.pad(logmel, ((0,0),(0,pad)), mode='constant')
    else:
        logmel = logmel[:, :MAX_LEN]

    return logmel[..., np.newaxis]

# ======================
# СБОР ДАННЫХ
# ======================
features, labels, tractors = [], [], []
print("Сканирование папок...")

for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    tractor_id = folder
    if "anomaly" in folder.lower() or "fault" in folder.lower():
        status = 1
    else:
        status = 0

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.wav', '.m4a')):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            X = extract_logmel(fpath, augment=False)
            features.append(X)
            labels.append(status)
            tractors.append(tractor_id)
        except Exception as e:
            print(f"⚠️ Ошибка с файлом {fpath}: {e}")

features = np.array(features, dtype=np.float32)
labels = np.array(labels)
tractors = np.array(tractors)
print(f"Всего записей: {len(features)} | норм: {(labels==0).sum()} | аномалий: {(labels==1).sum()}")

# ======================
# МОДЕЛЬ CRNN (улучшенная)
# ======================
def build_crnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # reshape для RNN
    x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# ======================
# K-FOLD ВАЛИДАЦИЯ
# ======================
gkf = GroupKFold(n_splits=N_SPLITS)
all_f1 = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(features, labels, groups=tractors)):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    model = build_crnn(X_train.shape[1:])

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6, verbose=1)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=1,
        callbacks=cb
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    f1 = f1_score(y_test, y_pred)
    all_f1.append(f1)

    print(f"F1-score: {f1:.3f}")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

print("\n========== ИТОГ ==========")
print(f"Средний F1 по {N_SPLITS} фолдам: {np.mean(all_f1):.3f} ± {np.std(all_f1):.3f}")

# ======================
# ФИНАЛЬНОЕ ОБУЧЕНИЕ И СОХРАНЕНИЕ
# ======================
print("\nОбучаем финальную модель на всех данных...")

final_model = build_crnn(features.shape[1:])
final_model.fit(features, labels, epochs=40, batch_size=16, verbose=1, callbacks=[
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
])

final_model.save("tractor_crnn_v2.h5")
joblib.dump({'sr': SR, 'n_mels': N_MELS, 'max_len': MAX_LEN}, "tractor_config.pkl")

print("✅ Модель сохранена: tractor_crnn_v2.h5")
