# train_best_panns_model_save_fix.py
import os, random, numpy as np, librosa, joblib
from tqdm import tqdm
from panns_inference import AudioTagging
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import csv

# ========== настройки ==========
SR = 16000
MAX_SEC = 6.0
DATA_DIR = "dataset"
MODELS_DIR = "models"
N_SPLITS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED)

os.makedirs(MODELS_DIR, exist_ok=True)

# ========== утилиты ==========
def get_label(folder):
    fn = folder.lower()
    return 1 if ("anomaly" in fn or "fault" in fn) else 0

def load_audio(path, sr=SR, max_sec=MAX_SEC):
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * max_sec)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def extract_panns_embedding(model, filepath):
    y = load_audio(filepath)
    with torch.no_grad():
        out = model.inference(torch.tensor(y[None, :]).to(DEVICE))
    # разные API -> универсальный обработчик
    emb = None
    if isinstance(out, dict):
        emb = out.get("embedding", out.get("clipwise_output", None))
    elif isinstance(out, (tuple, list)):
        emb = out[1] if len(out) > 1 else out[0]
    elif isinstance(out, np.ndarray):
        emb = out
    else:
        raise TypeError(f"Неожиданный тип out: {type(out)}")
    # привести к numpy
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb)
    emb = np.squeeze(emb)
    return emb

# ========== main ==========
def main():
    print("Загружаем PANNs...")
    model = AudioTagging(checkpoint_path=None, device=DEVICE)

    items = []
    for folder in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        label = get_label(folder)
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".m4a")):
                continue
            items.append({"path": os.path.join(folder_path, fname), "tractor": folder, "label": label})

    print("Всего файлов для обработки:", len(items))
    if len(items) == 0:
        raise SystemExit("Нет аудио файлов в dataset/ — положи туда данные.")

    features = []
    labels = []
    tractors = []
    bad_files = []

    for it in tqdm(items, desc="Extract embeddings"):
        p = it["path"]
        try:
            emb = extract_panns_embedding(model, p)
            if emb is None or emb.size == 0:
                bad_files.append((p, "empty embedding"))
                continue
            features.append(emb)
            labels.append(it["label"])
            tractors.append(it["tractor"])
        except Exception as e:
            bad_files.append((p, str(e)))

    features = np.array(features)
    labels = np.array(labels)
    tractors = np.array(tractors)

    # логируем плохие файлы
    if bad_files:
        bad_path = os.path.join(MODELS_DIR, "bad_files.csv")
        with open(bad_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "error"])
            writer.writerows(bad_files)
        print(f"⚠️ Были ошибки при извлечении эмбеддингов. Список -> {bad_path}")
        print("Первые 10 ошибок:")
        for bf in bad_files[:10]:
            print(" -", bf)

    print(f"✅ Успешных эмбеддингов: {len(features)}")

    if len(features) == 0:
        raise SystemExit("Ошибка: не удалось извлечь ни одного эмбеддинга. Проверь тест-скрипт test_extract_single.py")

    # сохраняем эмбеддинги и мета для удобства
    np.save(os.path.join(MODELS_DIR, "embeddings.npy"), features)
    np.save(os.path.join(MODELS_DIR, "labels.npy"), labels)
    np.save(os.path.join(MODELS_DIR, "tractors.npy"), tractors)
    print("Сохранены: models/embeddings.npy, labels.npy, tractors.npy")

    # масштабирование
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # KFold по тракторам
    gkf = GroupKFold(n_splits=min(N_SPLITS, max(2, len(features))))
    results = {"LR": [], "XGB": []}
    fold = 0
    for train_idx, test_idx in gkf.split(X, labels, groups=tractors):
        fold += 1
        print(f"\n=== Fold {fold} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # LR
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        f1_lr = f1_score(y_test, y_pred_lr)
        results["LR"].append(f1_lr)

        # XGB
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        results["XGB"].append(f1_xgb)

        print("LR F1:", f1_lr, "XGB F1:", f1_xgb)
        print(classification_report(y_test, y_pred_xgb, digits=3, zero_division=0))

    print("\nИтог:")
    print("LR mean f1:", np.mean(results["LR"]), "±", np.std(results["LR"]))
    print("XGB mean f1:", np.mean(results["XGB"]), "±", np.std(results["XGB"]))

    # выбираем лучшую модель по среднему f1
    mean_lr = np.mean(results["LR"])
    mean_xgb = np.mean(results["XGB"])
    best = "XGB" if mean_xgb > mean_lr else "LR"
    print("Лучшая модель:", best)

    # обучаем финальную модель на всех данных и сохраняем обе
    final_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    final_lr.fit(X, labels)
    final_xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
    final_xgb.fit(X, labels)

    joblib.dump(final_lr, os.path.join(MODELS_DIR, "panns_lr.joblib"))
    joblib.dump(final_xgb, os.path.join(MODELS_DIR, "panns_xgb.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "panns_scaler.joblib"))

    print("Сохранены модели в models/: panns_lr.joblib, panns_xgb.joblib, panns_scaler.joblib")
    print("Готово.")

if __name__ == "__main__":
    main()
