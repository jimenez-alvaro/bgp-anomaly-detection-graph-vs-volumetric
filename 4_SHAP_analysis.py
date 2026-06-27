"""
SHAP explainability analysis for Random Forest trained on volumetric and
topological BGP feature sets.

Input:
    dataset_csv_grande/volumen_rrc04_final.csv
    dataset_csv_grande/grafos_rrc04_final.csv

Output (saved to shap_outputs/):
    shap_barras_volumetrico.pdf   — Global feature importance bar plot
    shap_puntos_volumetrico.pdf   — Per-sample impact dot plot
    shap_barras_topologico.pdf    — Global feature importance bar plot
    shap_puntos_topologico.pdf    — Per-sample impact dot plot

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import re
import os

plt.rcParams['font.family'] = 'sans-serif'

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CSV_FOLDER   = "dataset_csv_grande/"
FILE_VOL     = "volumen_rrc04_final.csv"
FILE_GRAPH   = "grafos_rrc04_final.csv"
OUTPUT_DIR   = "shap_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def apply_log_and_zscore(df, cols_X):
    """
    1. Logarithmic Transformation to squash heavy right tails.
    2. Event-isolated Z-score normalization. The scaler is fitted exclusively 
       on normal (baseline) windows of each incident and applied to all windows.
    """
    df = df.copy()
    
    # 1. Logarithmic Transformation
    for col in cols_X:
        min_val = df[col].min()
        if min_val < 0:
            df[col] = np.log1p(df[col] - min_val)
        else:
            df[col] = np.log1p(df[col])
            
    # 2. Event-isolated Z-score normalization
    for base in df['Incidente_Base'].unique():
        idx_all    = df['Incidente_Base'] == base
        idx_normal = idx_all & (df['Label'] == 0)
        
        if idx_normal.sum() == 0:
            continue
            
        scaler = StandardScaler()
        scaler.fit(df.loc[idx_normal, cols_X])
        df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])
        
    return df


def generate_shap_figures(shap_values, X_test, paradigm_name):
    """
    Generate and save a bar plot (global feature importance) and a dot plot
    (per-sample feature impact) for a given paradigm.
    """
    # --- Bar plot: global feature importance ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_title(f"Global Feature Importance (Random Forest) {paradigm_name}", fontsize=13)
    ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_barras_{paradigm_name}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Bar plot saved: {path}")
    plt.close()

    # --- Dot plot: per-sample feature impact ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    ax = plt.gca()
    ax.set_title(f"Feature Impact on Anomaly Predictions (Random Forest) {paradigm_name}", fontsize=13)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    try:
        cbar = plt.gcf().axes[-1]
        cbar.set_ylabel("Feature value", fontsize=10)
        cbar.set_yticks([0, 1])
        cbar.set_yticklabels(["Low", "High"], fontsize=9)
    except Exception:
        pass
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"shap_puntos_{paradigm_name}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Dot plot saved: {path}")
    plt.close()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def run_shap(csv_file, paradigm_name):
    """
    Load a feature CSV, apply Log + event-isolated Z-score normalization, train
    Random Forest on a LOEO-consistent split (90% train / 10% test incidents),
    compute SHAP values, and save figures.
    """
    print(f"\n{'='*65}")
    print(f"SHAP — {paradigm_name.upper()}")
    print(f"{'='*65}")

    df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
    df['Incidente_Base'] = df['Evento'].apply(
        lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x))

    # Feature columns: exclude all metadata
    drop_cols = ['Timestamp', 'Evento', 'Incidente_Base', 'Label', 'Colector', 'Categoria']
    cols_X = df.drop(columns=[c for c in drop_cols if c in df.columns]).columns.tolist()

    # Preprocessing: Log + Event-isolated Z-score
    print("  Applying logarithmic transformation and event-isolated Z-score...")
    df = apply_log_and_zscore(df, cols_X)

    # LOEO-consistent train/test split: 90% train incidents, 10% test incidents
    unique_incidents = df['Incidente_Base'].unique()
    n_total = len(unique_incidents)

    np.random.seed(42)
    test_incidents = np.random.choice(
        unique_incidents,
        size=max(1, n_total // 10),
        replace=False
    )

    mask_test  = df['Incidente_Base'].isin(test_incidents)
    mask_train = ~mask_test

    X_train = df.loc[mask_train, cols_X]
    y_train = df.loc[mask_train, 'Label']
    X_test  = df.loc[mask_test,  cols_X]
    y_test  = df.loc[mask_test,  'Label']

    print(f"  Train incidents: ~{mask_train.sum() // 37} | "
          f"Test incidents: {len(test_incidents)}")
    print(f"  Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # Train Random Forest
    print("  Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()
    print(f"  Test accuracy: {acc:.4f}")

    # Compute SHAP values using TreeExplainer
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(X_test)
    
    # Random Forest returns a list of SHAP values (one array per class)
    # We select index 1 to explain the positive class (Anomaly)
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[1]
    elif len(shap_values_all.shape) == 3:
        shap_values = shap_values_all[:, :, 1]
    else:
        shap_values = shap_values_all

    # Save figures
    print("  Generating figures...")
    generate_shap_figures(shap_values, X_test, paradigm_name)

    # Print top 10 features by mean absolute SHAP value
    importances = np.abs(shap_values).mean(axis=0)
    top10 = pd.Series(importances, index=cols_X).sort_values(ascending=False).head(10)
    print(f"\n  Top 10 features by SHAP importance:")
    for feat, val in top10.items():
        print(f"    {feat:<35} {val:.4f}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    run_shap(FILE_VOL,   "volumetric")
    run_shap(FILE_GRAPH, "topological")
    print(f"\nSHAP analysis complete. Figures saved to: {OUTPUT_DIR}")