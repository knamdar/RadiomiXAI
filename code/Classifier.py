#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM + Perturbation-based importance for volume-confounded features (BN-aware).
Author: Ernest Namdar
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import bnlearn as bn

# === Step 0: Load BN model and extract node features ===
bn_model = bn.load("bayesian_network_model.json.pkl")
bn_features = list(bn_model["model"].nodes())


# === Step 1: Load and preprocess the dataset ===
csv_path = "../data/BraTS2020/Radiomics_binWidth-25_ImgHistNormalization_NETnNCR_T1CE.csv"
df = pd.read_csv(csv_path)

drop_cols = [
    "Patient_ID", "Group", "binWidth", "Normalization", "Survival_days",
    "Extent_of_Resection", "Subregion", "Sequence", "Age"
] + [col for col in df.columns if col.startswith("diagnostics")]
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=["Group_label"])
y = df["Group_label"]

# Keep only BN features in X
X = X[[col for col in X.columns if col in bn_features]]

# === Step 2: Train/Validation/Test split ===
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# === Step 3: Train LightGBM with GridSearchCV ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.05, 0.1]
}
clf = GridSearchCV(
    lgb.LGBMClassifier(random_state=0),
    param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# === Step 4: Evaluate ===
best_model = clf.best_estimator_
val_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"Best hyperparameters: {clf.best_params_}")
print(f"Validation AUC: {val_auc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# === Step 5: Perturbation-based importance function ===
def perturbation_importance(model, X_train, X_val, feature_names, baseline_preds=None, n_steps=30):
    if baseline_preds is None:
        baseline_preds = model.predict_proba(X_val)[:, 1]

    importance_scores = {}

    for feature in tqdm(feature_names, desc="Perturbation Importance"):
        fmin, fmax = X_train[feature].min(), X_train[feature].max()
        values = np.linspace(fmin, fmax, n_steps)

        sampled_values = np.random.choice(values, size=X_val.shape[0], replace=True)
        X_perturbed = X_val.copy()
        X_perturbed[feature] = sampled_values

        preds_perturbed = model.predict_proba(X_perturbed)[:, 1]
        diff = np.abs(preds_perturbed - baseline_preds).mean()
        importance_scores[feature] = diff

    return pd.Series(importance_scores).sort_values(ascending=False)

# === Step 6: Compute perturbation-based importance for confounded features ===
focus_features = ["original_shape_VoxelVolume", "original_shape_MeshVolume"]
val_preds = best_model.predict_proba(X_val)[:, 1]

importance_series = perturbation_importance(
    best_model, X_train, X_val, feature_names=focus_features, baseline_preds=val_preds
)

print("\nPerturbation-based importance for volume-confounded features:")
for f in focus_features:
    print(f"- {f}: {importance_series[f]:.6f}")


import matplotlib.pyplot as plt

bn_features = [f for f in bn_features if f in X_val.columns]  # keep only available columns

print(f"\nBN features found in validation set: {len(bn_features)}")

# === Step 7: Compute perturbation-based importance for all BN features ===
val_preds = best_model.predict_proba(X_val)[:, 1]

importance_series = perturbation_importance(
    best_model, X_train, X_val, feature_names=bn_features, baseline_preds=val_preds
)

# === Step 8: Tufte-style plot of top 5 BN feature importances ===
top_5 = importance_series.head(5)

plt.figure(figsize=(6, 3))
ax = top_5.plot(
    kind="barh",
    color="black",  # black ink, consistent with Tufte
    edgecolor="none"
)

# Invert y-axis for top-to-bottom ranking
ax.invert_yaxis()

# Minimalist labels
plt.xlabel("Importance", fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Remove chartjunk
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

plt.grid(False)
plt.title("")  # no title in Tufte style
plt.tight_layout()
plt.show()

# === Step 9: Compute uncertainty bounds for two target features using bn_model directly ===

target_features = ["original_shape_VoxelVolume", "original_shape_MeshVolume"]

# Extract edges and weights directly from bn_model
model_edges = bn_model["model_edges"]
from sklearn.metrics import mutual_info_score

# Recompute MI for each BN edge using training+val data
X_for_mi = X_trainval[bn_features]
edge_weights = {}

for (src, dst) in bn_model["model_edges"]:
    # Discretize if needed — assumes already binned features
    mi = mutual_info_score(X_for_mi[src], X_for_mi[dst])
    edge_weights[(src, dst)] = mi
print("\nUncertainty intervals using BN mutual information and perturbation-based importance:\n")

for target in bn_features: #target_features:
    if target not in importance_series:
        print(f"- {target} not in importance scores.\n")
        continue

    base_score = importance_series[target]

    # === Incoming: sum importance[src] * MI(src → target)
    incoming_sum = 0.0
    for (src, dst), mi in edge_weights.items():
        if dst == target and src in importance_series:
            incoming_sum += importance_series[src] * mi

    # === Outgoing: sum base_score * MI(target → dst)
    outgoing_sum = 0.0
    for (src, dst), mi in edge_weights.items():
        if src == target:
            outgoing_sum += base_score * mi

    lower = base_score - incoming_sum
    upper = base_score + outgoing_sum

    print(f"{target}:")
    print(f"  Base Importance     : {base_score:.6f}")
    print(f"  Incoming Adjustment : -{incoming_sum:.6f}")
    print(f"  Outgoing Adjustment : +{outgoing_sum:.6f}")
    print(f"  Confidence Interval : [{lower:.6f}, {upper:.6f}]\n")


# === Step X: Compute confidence intervals for all BN features ===
importance_ci = []

for target in bn_features:
    if target not in importance_series:
        continue

    base_score = importance_series[target]

    incoming_sum = sum(
        importance_series.get(src, 0) * mi
        for (src, dst), mi in edge_weights.items()
        if dst == target and src in importance_series
    )
    outgoing_sum = sum(
        base_score * mi
        for (src, dst), mi in edge_weights.items()
        if src == target
    )

    lower = base_score - incoming_sum
    upper = base_score + outgoing_sum
    importance_ci.append({
        "feature": target,
        "importance": base_score,
        "lower": lower,
        "upper": upper,
        "error": (upper - lower) / 2  # symmetric approximation
    })

# Convert to DataFrame
ci_df = pd.DataFrame(importance_ci)
top5_ci = ci_df.sort_values("importance", ascending=False).head(5).copy()
top5_ci = top5_ci[::-1]  # reverse for top-to-bottom plotting

# === Step Y: Tufte-style horizontal error bar plot ===
plt.figure(figsize=(6, 3))
plt.errorbar(
    x=top5_ci["importance"],
    y=top5_ci["feature"],
    xerr=top5_ci["error"],
    fmt='o',
    color='black',
    ecolor='gray',
    capsize=3,
    markersize=5
)

plt.xlabel("Perturbation-Based Importance", fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Remove chartjunk
ax = plt.gca()
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)
plt.grid(False)
plt.title("")  # No title
plt.tight_layout()
plt.show()
