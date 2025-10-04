# fifa21_ml_benchmark_plus.py
# Decision Tree, Random Forest, GaussianNB, MLP (+SMOTE & tuned variants)
# Deep Learning (Keras MLP)

import os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# SMOTE pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Seeds
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED)

# -------------------------------
# 0) PATH TO DATA
# -------------------------------
CSV_PATH = "./players_21.csv"   # <-- change to your path if needed

# -------------------------------
# 1) LOAD DATA
# -------------------------------
def load_fifa(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "sofifa_id" in df.columns:
        df = df.drop_duplicates(subset=["sofifa_id"])
    return df

df = load_fifa(CSV_PATH)

if "overall" not in df.columns:
    raise ValueError("The dataset must contain an 'overall' column. Please verify you're using players_21.csv")

# -------------------------------
# 2) DEFINE TARGET (bucketed overall)
# -------------------------------
def bucket_overall(x):
    if x < 70: return "Low"
    if x < 80: return "Mid"
    return "High"

df["overall_tier"] = df["overall"].apply(bucket_overall)

# -------------------------------
# 3) SELECT FEATURES (avoid leakage)
# -------------------------------
drop_cols = {
    "overall","short_name","long_name","player_url","sofifa_id",
    "potential","club","nation_position","team_position","body_type",
    "real_face","player_positions","dob","joined","contract_valid_until",
    "team_jersey_number","nation_jersey_number","loaned_from"
}

numeric_candidates = [
    "age","height_cm","weight_kg","value_eur","wage_eur",
    "international_reputation","weak_foot","skill_moves",
    "pace","shooting","passing","dribbling","defending","physic",
    "attacking_crossing","attacking_finishing","attacking_heading_accuracy",
    "attacking_short_passing","attacking_volleys",
    "skill_dribbling","skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control",
    "movement_acceleration","movement_sprint_speed","movement_agility","movement_reactions","movement_balance",
    "power_shot_power","power_jumping","power_stamina","power_strength","power_long_shots",
    "mentality_aggression","mentality_interceptions","mentality_positioning","mentality_vision",
    "mentality_penalties","mentality_composure",
    "defending_marking","defending_standing_tackle","defending_sliding_tackle",
    "goalkeeping_diving","goalkeeping_handling","goalkeeping_kicking","goalkeeping_positioning","goalkeeping_reflexes"
]

cat_candidates = ["preferred_foot","work_rate","nationality"]

numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in cat_candidates if c in df.columns]

use_cols = numeric_features + categorical_features + ["overall_tier"]
data = df[use_cols].copy()

X = data.drop(columns=["overall_tier"])
y = data["overall_tier"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# -------------------------------
# 4) PREPROCESSING
# -------------------------------
numeric_common = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_common = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor_common = ColumnTransformer(
    transformers=[
        ("num", numeric_common, numeric_features),
        ("cat", categorical_common, categorical_features)
    ],
    remainder="drop"
)
preprocessor_common.set_output(transform="pandas")


# For MLP: scale after OHE
numeric_mlp = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True))
])

categorical_mlp = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("scaler", StandardScaler(with_mean=False))
])

preprocessor_mlp = ColumnTransformer(
    transformers=[
        ("num", numeric_mlp, numeric_features),
        ("cat", categorical_mlp, categorical_features)
    ],
    remainder="drop"
)
preprocessor_mlp.set_output(transform="pandas")


# -------------------------------
# 5) MODELS (baseline + advanced)
# -------------------------------
models = {}

# Baselines
models["DecisionTree"] = Pipeline(steps=[
    ("prep", preprocessor_common),
    ("clf", DecisionTreeClassifier(criterion="gini", max_depth=None,
                                   min_samples_split=4, random_state=SEED))
])

models["RandomForest"] = Pipeline(steps=[
    ("prep", preprocessor_common),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                   min_samples_split=4, n_jobs=-1,
                                   class_weight="balanced", random_state=SEED))
])

models["GaussianNB"] = Pipeline(steps=[
    ("prep", preprocessor_common),
    ("clf", GaussianNB())
])

# Tuned MLP (no early stopping to avoid string-label bug)
models["MLP_tuned"] = Pipeline(steps=[
    ("prep", preprocessor_mlp),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256,128,64),
        activation="relu", solver="adam",
        learning_rate_init=5e-4, alpha=5e-5,
        batch_size=256, max_iter=200,
        early_stopping=False,
        random_state=SEED, verbose=False))
])

# SMOTE variant for RF (remove class_weight to avoid over/over)
models["RandomForest_SMOTE"] = ImbPipeline(steps=[
    ("prep", preprocessor_common),
    ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=4,
        n_jobs=-1, random_state=SEED))
])

# Tuned RF
models["RandomForest_tuned"] = Pipeline(steps=[
    ("prep", preprocessor_common),
    ("clf", RandomForestClassifier(
        n_estimators=600, max_features="sqrt",
        min_samples_split=2, min_samples_leaf=1,
        max_depth=None, n_jobs=-1, random_state=SEED))
])

# -------------------------------
# 6) TRAIN & EVALUATE
# -------------------------------
os.makedirs("figures", exist_ok=True)
results = []

def evaluate_model(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)

    acc = accuracy_score(y_te, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)
    report = classification_report(y_te, preds, digits=3)
    cm = confusion_matrix(y_te, preds, labels=["Low","Mid","High"])

    print(f"\n=== {name} ===")
    print(report)

    # Confusion matrix plot
    labels = ["Low","Mid","High"]
    fig, ax = plt.subplots(figsize=(4.5,3.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(f"figures/confmat_{name}.png", dpi=160)
    plt.close(fig)

    return {"model": name, "accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1, "cm": cm, "pipeline": pipe}

for name, pipe in models.items():
    res = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
    results.append(res)

summary = pd.DataFrame([{
    "Model": r["model"],
    "Accuracy": round(r["accuracy"], 4),
    "Precision (macro)": round(r["precision_macro"], 4),
    "Recall (macro)": round(r["recall_macro"], 4),
    "F1 (macro)": round(r["f1_macro"], 4),
} for r in results]).sort_values("F1 (macro)", ascending=False)

print("\n==== SUMMARY (sorted by macro-F1) ====\n")
print(summary.to_string(index=False))
summary.to_csv("figures/model_summary.csv", index=False)

# -------------------------------
# 7) 5-FOLD CV (macro-F1)
# -------------------------------
def cv_macro_f1(pipe, X_df, y_ser, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, te_idx in skf.split(X_df, y_ser):
        X_tr, X_te = X_df.iloc[tr_idx], X_df.iloc[te_idx]
        y_tr, y_te = y_ser.iloc[tr_idx], y_ser.iloc[te_idx]
        est = pipe
        est.fit(X_tr, y_tr)
        preds = est.predict(X_te)
        f1 = precision_recall_fscore_support(y_te, preds, average="macro", zero_division=0)[2]
        scores.append(f1)
    return np.array(scores)

print("\n==== 5-fold CV macro-F1 (optional) ====")
for name, pipe in models.items():
    scores = cv_macro_f1(pipe, X, y, n_splits=5)
    print(f"{name}: mean={scores.mean():.4f}  std={scores.std():.4f}  folds={np.round(scores,4)}")

# ===============================
# 7b) Deep Learning (Keras MLP++)
# ===============================
try:
    # Try TF keras first, fallback to standalone keras
    try:
        from tensorflow import keras
        from keras import layers
    except Exception:
        from keras import layers, models as keras  # fallback

    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

    print("\n[DL] Preparing data for Keras...")
    # Reuse the SAME preprocessing as sklearn MLP (scaled dense matrix)
    X_train_proc = preprocessor_mlp.fit_transform(X_train)
    X_test_proc  = preprocessor_mlp.transform(X_test)

    # Encode y -> integers for softmax
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int  = le.transform(y_test)
    num_classes = len(le.classes_)  # should be 3 (Low/Mid/High)

    # Class weights to help minority class
    class_counts = np.bincount(y_train_int)
    total = class_counts.sum()
    class_weight = {i: (total / (num_classes * c)) for i, c in enumerate(class_counts)}

    # Keras model: BN + Dense + Dropout stacks
    def build_model(input_dim, num_classes):
        inputs = keras.Input(shape=(input_dim,))
        x = layers.BatchNormalization()(inputs)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)

        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs, name="Keras_DeepMLP")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    input_dim = X_train_proc.shape[1]
    model = build_model(input_dim, num_classes)

    # Callbacks & training (no checkpoint files)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
    ]

    print("\n[DL] Training Keras DeepMLP...")
    history = model.fit(
        X_train_proc, y_train_int,
        validation_split=0.15,
        epochs=200,
        batch_size=256,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0
    )
    print("[DL] Training complete.")

    # Plot training curves
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Keras DeepMLP — Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/keras_deepmlp_loss.png", dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Keras DeepMLP — Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/keras_deepmlp_acc.png", dpi=150); plt.close()

    # Evaluate
    print("[DL] Evaluating on test set...")
    y_pred_probs = model.predict(X_test_proc, verbose=0)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    y_pred = le.inverse_transform(y_pred_int)
    y_true = y_test.values  # strings

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, digits=3, labels=["Low","Mid","High"])
    cm = confusion_matrix(y_true, y_pred, labels=["Low","Mid","High"])
    print("\n=== Keras_DeepMLP ===")
    print(report)

    # Confusion matrix image (consistent style)
    labels = ["Low","Mid","High"]
    fig, ax = plt.subplots(figsize=(4.5,3.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix — Keras_DeepMLP")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig("figures/confmat_Keras_DeepMLP.png", dpi=160)
    plt.close(fig)

    # Append to your summary table and resave
    dl_row = pd.DataFrame([{
        "Model": "Keras_DeepMLP",
        "Accuracy": round(acc, 4),
        "Precision (macro)": round(prec, 4),
        "Recall (macro)": round(rec, 4),
        "F1 (macro)": round(f1, 4),
    }])
    summary = pd.concat([summary, dl_row], ignore_index=True).sort_values("F1 (macro)", ascending=False)
    print("\n==== SUMMARY + Keras_DeepMLP (sorted by macro-F1) ====\n")
    print(summary.to_string(index=False))
    summary.to_csv("figures/model_summary.csv", index=False)

except Exception as e:
    print("\n[DL] Skipping Keras model due to error:", e)

# -------------------------------
# 8) PERMUTATION FEATURE IMPORTANCE (RF best) — use TRANSFORMED X
# -------------------------------

def pretty_feature_names(names):
    out = []
    for s in map(str, names):
        s = s.replace("num__", "").replace("cat__", "").replace("onehot__", "")
        s = s.replace("preferred_foot_", "preferred_foot=")
        s = s.replace("work_rate_", "work_rate=")
        s = s.replace("nationality_", "nationality=")
        out.append(s)
    return np.array(out)

rf_like_preference = ["RandomForest_tuned", "RandomForest_SMOTE", "RandomForest", "DecisionTree"]
chosen_name = next((m for m in rf_like_preference if m in [r["model"] for r in results]), None)

if chosen_name is not None:
    # get fitted pipeline
    idx = [r["model"] for r in results].index(chosen_name)
    pipe = results[idx]["pipeline"]
    pipe.fit(X_train, y_train)

    # separate fitted preprocessor & classifier
    prep = pipe.named_steps["prep"]
    clf  = pipe.named_steps["clf"]

    # **transform X_test to the final feature space**
    Xt_test = prep.transform(X_test)           # returns DataFrame 
    feat_names = pretty_feature_names(np.asarray(Xt_test.columns))

    print(f"\n=== Permutation Importance on: {chosen_name} (transformed space) ===")

    # run PI on the classifier with transformed features
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        clf, Xt_test, y_test, n_repeats=10, random_state=SEED, n_jobs=-1
    )

    n_import = perm.importances_mean.shape[0]
    assert n_import == len(feat_names), f"mismatch: {n_import} vs {len(feat_names)}"

    imp_series = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)

    # save CSVs
    all_csv = f"figures/perm_importance_{chosen_name}.csv"
    top_csv = f"figures/perm_importance_{chosen_name}_TOP20.csv"
    (imp_series
        .reset_index()
        .rename(columns={"index":"feature", 0:"importance_mean"})
        .to_csv(all_csv, index=False))
    (imp_series.head(20)
        .reset_index()
        .rename(columns={"index":"feature", 0:"importance_mean"})
        .to_csv(top_csv, index=False))

    # plot top-k with trimmed labels
    topk = min(30, len(imp_series))
    trimmed = imp_series.head(topk).iloc[::-1]
    trimmed.index = [s if len(s) <= 40 else s[:37] + "..." for s in trimmed.index]

    fig, ax = plt.subplots(figsize=(7.5, 9))
    trimmed.plot(kind="barh", ax=ax)
    ax.set_title(f"Permutation Importance (Top {topk}) — {chosen_name}")
    ax.set_xlabel("Importance (mean decrease in score)")
    plt.tight_layout()
    fig.savefig(f"figures/perm_importance_{chosen_name}.png", dpi=170)
    plt.close(fig)
else:
    print("\n(No RF-like model found for permutation importance.)")


print("\nAll artifacts saved into ./figures")
