import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten,
    LSTM,
    Input,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ============================================================
# TrustV2X - Step 04
# Train Deep Learning Models: MLP, 1D-CNN, LSTM
# ============================================================

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "sumo_generated" / "trustv2x_attack_dataset.csv"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_MODEL_DIR = PROJECT_ROOT / "results" / "models"

RESULTS_FILE = RESULTS_TABLE_DIR / "dl_binary_results.csv"
REPORT_FILE = RESULTS_TABLE_DIR / "dl_classification_reports.txt"

RANDOM_SEED = 42
TEST_SIZE = 0.25
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def ensure_directories():
    for folder in [RESULTS_TABLE_DIR, RESULTS_FIGURE_DIR, RESULTS_MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run scripts/02_inject_attacks.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("TrustV2X - Deep Learning Training")
    print("=" * 70)

    print(f"[INFO] Dataset loaded: {DATA_FILE}")
    print(f"[INFO] Dataset shape: {df.shape}")
    print("\n[INFO] Attack type distribution:")
    print(df["attack_type"].value_counts())

    return df


def get_feature_columns(df):
    numeric_features = [
        "message_sequence",
        "timestamp",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "heading",
        "lane_position",
        "neighbour_count",
        "inter_arrival_time",
        "delta_t",
        "distance_from_previous",
        "estimated_speed_from_position",
        "speed_position_error",
        "heading_change",
        "speed_change",
        "message_rate_5s",
        "identity_cluster_count",
        "sender_message_count",
        "flag_unrealistic_speed",
        "flag_hard_acceleration",
        "flag_large_position_jump",
        "flag_speed_position_mismatch",
        "flag_heading_anomaly",
        "flag_stale_or_irregular_time",
        "flag_dos_rate",
        "flag_possible_sybil",
        "flag_low_sender_history",
        "flag_neighbour_anomaly",
        "trustv2x_rule_violation_count",
    ]

    categorical_features = [
        "lane_id",
        "road_id",
        "message_type",
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    return numeric_features, categorical_features

def preprocess_dataset(df):
    numeric_features, categorical_features = get_feature_columns(df)

    X = df[numeric_features + categorical_features]
    y = df["label"].astype(int).values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("\n[INFO] Feature preprocessing complete")
    print(f"[INFO] Numeric features: {numeric_features}")
    print(f"[INFO] Categorical features: {categorical_features}")
    print(f"[INFO] Processed feature shape: {X_processed.shape}")
    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, preprocessor


def build_mlp(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.30),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_1d_cnn(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Dropout(0.25),
            Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_lstm(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dropout(0.20),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def false_alarm_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


def detection_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm)

    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticklabels(["Normal", "Attack"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
    figure_path = RESULTS_FIGURE_DIR / f"cm_{safe_model_name}_dl.png"

    plt.savefig(figure_path, dpi=300)
    plt.close()

    return figure_path


def plot_training_curve(history, model_name):
    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["accuracy"], label="Training Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_title(f"{model_name} Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()

    acc_path = RESULTS_FIGURE_DIR / f"training_accuracy_{safe_model_name}.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title(f"{model_name} Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()

    loss_path = RESULTS_FIGURE_DIR / f"training_loss_{safe_model_name}.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return acc_path, loss_path


def evaluate_model(model, model_name, X_test, y_test, reshape_for_sequence=False):
    if reshape_for_sequence:
        X_eval = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    else:
        X_eval = X_test

    start_pred = time.perf_counter()
    y_prob = model.predict(X_eval, verbose=0).ravel()
    pred_time = time.perf_counter() - start_pred

    y_pred = (y_prob >= 0.5).astype(int)

    avg_latency_ms = (pred_time / len(X_test)) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    far = false_alarm_rate(y_test, y_pred)
    dr = detection_rate(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cm_path = plot_confusion_matrix(cm, model_name)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"],
        zero_division=0,
    )

    result_row = {
        "feature_set": "TrustV2X_Enhanced_Features",
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_detection_rate": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "detection_rate": round(dr, 4),
        "prediction_time_seconds": round(pred_time, 4),
        "avg_latency_ms_per_message": round(avg_latency_ms, 6),
        "confusion_matrix_file": str(cm_path),
    }

    report_text = (
        f"\n{'=' * 70}\n"
        f"Model: {model_name}\n"
        f"{'=' * 70}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{report}\n"
    )

    return result_row, report_text


def train_deep_learning_models(X_train, X_test, y_train, y_test):
    input_dim = X_train.shape[1]

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    results = []
    reports = []

    # -------------------------------
    # MLP
    # -------------------------------
    print("\n[TRAINING] MLP")

    mlp = build_mlp(input_dim)

    start_train = time.perf_counter()
    mlp_history = mlp.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    mlp_train_time = time.perf_counter() - start_train

    mlp_path = RESULTS_MODEL_DIR / "mlp_trustv2x_enhanced_features.keras"
    mlp.save(mlp_path)

    plot_training_curve(mlp_history, "MLP")

    mlp_result, mlp_report = evaluate_model(
        mlp,
        "MLP",
        X_test,
        y_test,
        reshape_for_sequence=False,
    )

    mlp_result["training_time_seconds"] = round(mlp_train_time, 4)
    mlp_result["model_file"] = str(mlp_path)

    results.append(mlp_result)
    reports.append(mlp_report)

    print(
        f"[RESULT] MLP | Acc={mlp_result['accuracy']}, "
        f"Prec={mlp_result['precision']}, "
        f"Recall={mlp_result['recall_detection_rate']}, "
        f"F1={mlp_result['f1_score']}, "
        f"FAR={mlp_result['false_alarm_rate']}, "
        f"Latency={mlp_result['avg_latency_ms_per_message']} ms"
    )

    # Prepare feature-vector-as-sequence for 1D-CNN and LSTM
    X_train_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_seq = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # -------------------------------
    # 1D-CNN
    # -------------------------------
    print("\n[TRAINING] 1D-CNN")

    cnn = build_1d_cnn((input_dim, 1))

    start_train = time.perf_counter()
    cnn_history = cnn.fit(
        X_train_seq,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    cnn_train_time = time.perf_counter() - start_train

    cnn_path = RESULTS_MODEL_DIR / "1d_cnn_trustv2x_enhanced_features.keras"
    cnn.save(cnn_path)

    plot_training_curve(cnn_history, "1D-CNN")

    cnn_result, cnn_report = evaluate_model(
        cnn,
        "1D-CNN",
        X_test,
        y_test,
        reshape_for_sequence=True,
    )

    cnn_result["training_time_seconds"] = round(cnn_train_time, 4)
    cnn_result["model_file"] = str(cnn_path)

    results.append(cnn_result)
    reports.append(cnn_report)

    print(
        f"[RESULT] 1D-CNN | Acc={cnn_result['accuracy']}, "
        f"Prec={cnn_result['precision']}, "
        f"Recall={cnn_result['recall_detection_rate']}, "
        f"F1={cnn_result['f1_score']}, "
        f"FAR={cnn_result['false_alarm_rate']}, "
        f"Latency={cnn_result['avg_latency_ms_per_message']} ms"
    )

    # -------------------------------
    # LSTM
    # -------------------------------
    print("\n[TRAINING] LSTM")

    lstm = build_lstm((input_dim, 1))

    start_train = time.perf_counter()
    lstm_history = lstm.fit(
        X_train_seq,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    lstm_train_time = time.perf_counter() - start_train

    lstm_path = RESULTS_MODEL_DIR / "lstm_trustv2x_enhanced_features.keras"
    lstm.save(lstm_path)

    plot_training_curve(lstm_history, "LSTM")

    lstm_result, lstm_report = evaluate_model(
        lstm,
        "LSTM",
        X_test,
        y_test,
        reshape_for_sequence=True,
    )

    lstm_result["training_time_seconds"] = round(lstm_train_time, 4)
    lstm_result["model_file"] = str(lstm_path)

    results.append(lstm_result)
    reports.append(lstm_report)

    print(
        f"[RESULT] LSTM | Acc={lstm_result['accuracy']}, "
        f"Prec={lstm_result['precision']}, "
        f"Recall={lstm_result['recall_detection_rate']}, "
        f"F1={lstm_result['f1_score']}, "
        f"FAR={lstm_result['false_alarm_rate']}, "
        f"Latency={lstm_result['avg_latency_ms_per_message']} ms"
    )

    return results, reports


def main():
    ensure_directories()

    df = load_dataset()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_dataset(df)

    results, reports = train_deep_learning_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["f1_score", "accuracy"],
        ascending=[False, False],
    )

    results_df.to_csv(RESULTS_FILE, index=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(reports))

    print("\n" + "=" * 70)
    print("[OK] Deep learning training complete")
    print("=" * 70)
    print(f"[OK] Results saved to: {RESULTS_FILE}")
    print(f"[OK] Reports saved to: {REPORT_FILE}")
    print(f"[OK] Models saved to: {RESULTS_MODEL_DIR}")

    print("\nTop deep learning results:")
    print(results_df.to_string(index=False))

    print("\nDone. Next step: TrustV2X scoring using 05_trustv2x_scoring.py")


if __name__ == "__main__":
    main()