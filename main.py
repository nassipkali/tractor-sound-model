import os
import random
import numpy as np
import librosa
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

# ======================
# НАСТРОЙКИ
# ======================
SR = 16000
N_MFCC = 40
N_SPLITS = 5
AUGMENT_PROB = 0.3
DATA_DIR = "dataset"  # Корневая папка

# ======================
# АУГМЕНТАЦИИ
# ======================
def add_noise(y, noise_factor=0.02):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return np.clip(augmented, -1.0, 1.0)

def time_shift(y, shift_max=0.2):
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps_range=(-2, 2)):
    n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def stretch_audio(y, rate_range=(0.9, 1.1)):
    """Безопасная версия растяжки времени"""
    rate = random.uniform(rate_range[0], rate_range[1])
    return librosa.resample(y, orig_sr=SR, target_sr=int(SR * rate))


# ======================
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ======================
def extract_features(file_path, sr=SR, n_mfcc=N_MFCC, augment=False):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    y = y / np.max(np.abs(y))  # нормализация амплитуды

    # --- Аугментации ---
    if augment:
        if random.random() < AUGMENT_PROB: 
            y = add_noise(y)
        if random.random() < AUGMENT_PROB: 
            y = time_shift(y)
        if random.random() < AUGMENT_PROB: 
            y = pitch_shift(y, sr)
        if random.random() < AUGMENT_PROB: 
            y = stretch_audio(y)   # ✅ ← исправленный вызов

    # --- MFCC признаки ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.median(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])
    return features


# ======================
# СБОР ДАННЫХ ИЗ ПАПОК
# ======================
features, labels, tractors = [], [], []

print("Сканирование папок...")

for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    # Определяем класс и ID трактора по названию папки
    tractor_id = folder
    if "anomaly" in folder.lower():
        status = 1
    elif "fault" in folder.lower():
        status = 1
    else:
        status = 0

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.wav', '.m4a')):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            X = extract_features(fpath, augment=False)
            features.append(X)
            labels.append(status)
            tractors.append(tractor_id)
        except Exception as e:
            print(f"⚠️ Ошибка с файлом {fpath}: {e}")

features = np.array(features)
labels = np.array(labels)
tractors = np.array(tractors)

print(f"Всего записей: {len(features)} | норм: {(labels==0).sum()} | аномалий: {(labels==1).sum()}")

# ======================
# K-FOLD ВАЛИДАЦИЯ ПО ТРАКТОРАМ
# ======================
gkf = GroupKFold(n_splits=N_SPLITS)
all_f1 = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(features, labels, groups=tractors)):
    print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")

    # --- Извлекаем train с аугментациями ---
    X_train, y_train = [], []
    for i in train_idx:
        folder = tractors[i]
        file_list = [f for f in os.listdir(os.path.join(DATA_DIR, folder)) if f.endswith(('.wav', '.m4a'))]
        # случайный файл, чтобы чуть разнообразить
        sample_file = os.path.join(DATA_DIR, folder, random.choice(file_list))
        X_train.append(extract_features(sample_file, augment=True))
        y_train.append(labels[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test, y_test = features[test_idx], labels[test_idx]

    # --- Масштабирование ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Модель XGBoost ---
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Предсказания и метрики ---
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    all_f1.append(f1)

    print(f"F1-score: {f1:.3f}")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

print("\n========== ИТОГ ==========")
print(f"Средний F1 по {N_SPLITS} фолдам: {np.mean(all_f1):.3f} ± {np.std(all_f1):.3f}")

print("\nОбучаем финальную модель на всех данных...")

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# масштабирование на всех данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# финальная модель
final_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_scaled, labels)

# сохраняем модель и scaler
joblib.dump(final_model, "tractor_model.pkl")
joblib.dump(scaler, "tractor_scaler.pkl")

print("✅ Модель и scaler сохранены:")
print(" - tractor_model.pkl")
print(" - tractor_scaler.pkl")