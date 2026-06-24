"""
bgp_confusion_matrices.py
=========================
Generates cumulative LOEO-CV confusion matrices for XGBoost under both
volumetric and topological feature paradigms.

Each LOEO iteration holds out one complete incident (normal + attack windows),
trains XGBoost on the remaining incidents, and accumulates predictions.
The final confusion matrix is built from all accumulated predictions across
all iterations, giving a global picture of model errors.

Input:
    dataset_csv_grande/volumen_rrc04_final.csv
    dataset_csv_grande/grafos_rrc04_final.csv

Output (saved to confusion_matrix_outputs/):
    matriz_confusion_volumetric_rrc04_final.pdf
    matriz_confusion_topological_rrc04_final.pdf

Usage:
    python bgp_confusion_matrices.py
"""

import pandas as pd
import numpy as np
import re
import os

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CSV_FOLDER   = "dataset_csv_grande/"
CSV_FILES    = ["volumen_rrc04_final.csv",   "grafos_rrc04_final.csv"]
PLOT_NAMES   = ["Volumetric",                "Topological"]
OUTPUT_DIR   = "confusion_matrix_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def apply_zscore(df, cols_X):
    """
    Event-isolated Z-score normalization.
    The scaler is fitted exclusively on normal (baseline) windows of each
    incident and applied to all windows of that incident.
    This removes the Year Effect without leaking information across incidents.
    """
    df = df.copy()
    for base in df['Incidente_Base'].unique():
        idx_all    = df['Incidente_Base'] == base
        idx_normal = idx_all & (df['Label'] == 0)
        if idx_normal.sum() == 0:
            continue
        scaler = StandardScaler()
        scaler.fit(df.loc[idx_normal, cols_X])
        df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])
    return df


def run_loeo(df, cols_X):
    """
    Run the full LOEO-CV loop for XGBoost and return accumulated
    true labels and predictions across all iterations.
    """
    X         = df[cols_X].values
    y         = df['Label'].values
    incidents = df['Incidente_Base'].values
    unique    = np.unique(incidents)

    y_true_all, y_pred_all = [], []

    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)

    for i, incident_test in enumerate(unique):
        mask_test  = incidents == incident_test
        mask_train = ~mask_test

        X_train, y_train = X[mask_train], y[mask_train]
        X_test,  y_test  = X[mask_test],  y[mask_test]

        # Skip if test set contains only one class
        if len(np.unique(y_test)) < 2:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(unique)} incidents processed...")

    return np.array(y_true_all), np.array(y_pred_all), len(unique)


def plot_confusion_matrix(cm, paradigm_name, n_incidents, output_path):
    """
    Plot and save the confusion matrix as a PDF.
    """
    tn, fp, fn, tp = cm.ravel()
    labels = ['Normal (0)', 'Anomaly (1)']

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 14}
    )
    plt.title(
        f'Confusion Matrix XGBoost — {paradigm_name} Paradigm\n'
        f'(Leave-One-Event-Out CV | {n_incidents} incidents)',
        fontsize=13, pad=15
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label (Ground Truth)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    for csv_file, paradigm_name in zip(CSV_FILES, PLOT_NAMES):
        print(f"\n{'='*65}")
        print(f"Confusion matrix — {paradigm_name} ({csv_file})")
        print(f"{'='*65}")

        # Load and preprocess
        df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
        df['Incidente_Base'] = df['Evento'].apply(
            lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x))

        drop_cols = ['Timestamp', 'Evento', 'Incidente_Base', 'Label', 'Colector', 'Categoria']
        cols_X = df.drop(columns=[c for c in drop_cols if c in df.columns]).columns.tolist()

        print(f"  Unique incidents: {df['Incidente_Base'].nunique()} | "
              f"Total rows: {len(df)}")

        df = apply_zscore(df, cols_X)

        # LOEO-CV
        y_true, y_pred, n_incidents = run_loeo(df, cols_X)

        # Metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Global LOEO results:")
        print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"    Accuracy  = {(tp+tn)/(tp+tn+fp+fn):.4f}")
        print(f"    Recall    = {tp/(tp+fn):.4f}")
        print(f"    Precision = {tp/(tp+fp):.4f}")

        # Save figure
        output_path = os.path.join(
            OUTPUT_DIR,
            f"matriz_confusion_{paradigm_name.lower()}_rrc04_final.pdf"
        )
        plot_confusion_matrix(cm, paradigm_name, n_incidents, output_path)

    print(f"\nConfusion matrices complete. PDFs saved to: {OUTPUT_DIR}")