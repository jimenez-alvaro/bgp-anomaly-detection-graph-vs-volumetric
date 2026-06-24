"""
Leave-One-Event-Out (LOEO) cross-validation for BGP anomaly detection.
Trains and evaluates four classifiers on volumetric and topological feature sets.

Input:
    dataset_csv_grande/volumen_rrc04_final.csv   — 32 volumetric features
    dataset_csv_grande/grafos_rrc04_final.csv    — 14 topological features

Output:
    resultados_LOEO_final.csv   — Global and per-category metrics for all models

Evaluation protocol:
    In each of N iterations (one per unique incident), the complete incident
    (all normal and attack windows) is held out as the test set. The model is
    trained on the remaining N-1 incidents. This guarantees the model is always
    evaluated on incidents it has never seen during training, faithfully
    simulating real operational deployment.

Normalization:
    Event-isolated Z-score: the scaler is fitted exclusively on the normal
    (baseline) windows of each incident and then applied to all windows of
    that incident. This removes the Year Effect (Internet growth over 19 years)
    without leaking information between train and test sets.

Usage:
    python bgp_loeo_training.py
"""

import pandas as pd
import numpy as np
import re
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CSV_FOLDER = "dataset_csv_grande/"
CSV_FILES  = ["volumen_rrc04_final.csv", "grafos_rrc04_final.csv"]

# ==============================================================================
# INCIDENT CATEGORY MAPPING
# ==============================================================================

def get_category(event_name):
    """
    Assign an anomaly category to a row based on the event name.
    Both normal and attack windows of the same incident receive the same category,
    derived from the base incident name (suffix stripped).
    """
    base = re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', event_name)

    hijack_bases = [
        "YouTube", "Amazon_MEW", "Rostelecom", "Twitter", "CelerNetwork",
        "ChinaTelecom2010", "Beltelecom2013", "HackingTeam2013", "Indosat2014",
        "Airtel2015", "eNet_Google2018", "Telstra2019", "Lumen2020",
        "AS209306_2021", "VodafoneAS55410", "CERNET2021", "TurkTelecom2022",
        "AS267613_2022", "MangoDSL2022", "Cobranet2023", "Eletronet2023",
        "MyRepublic2023", "RCSRDS2023", "Digicel2023", "UzbekAS2023", "PTCL2023",
        "Bitcoin_AS32653", "Visa_Rostelecom", "Bitcanal", "ChinaTelecom",
        "Swisscom", "Renesys2008", "GlobalCrossing2010", "TelecomItalia2011",
        "Rostelecom2017", "Level3_2017", "Backconnect2018", "EthereumClassic2019",
        "Maxis2019", "Telekomunikacja2019", "M247_2020", "Windstream2020",
        "TATA2021", "Zayo2021", "NTT2021", "Rostelecom2022", "Cogent2022",
        "ChinaMobile2022", "Turkcell2022", "Bharti2023", "RETN2023",
        "Zayo2023", "Lumen2023", "MTS2023",
    ]

    leak_bases = [
        "TelekomMalaysia", "Google_MainOne", "Cloudflare", "CenturyLink",
        "Vodafone_India", "TurkTelecom2004", "AGIS2010", "ChinaTelecom2010Leak",
        "Moratel2012", "Dodo2014", "Hathway2015", "Telstra2016", "Level3_2017_Leak",
        "Brazil2017", "Japan2017", "Verizon2019", "Telus2020", "CoreBackbone2020",
        "Iliad2021", "TelekomSerbia2021", "Guangdong2021", "Telstra2022",
        "Altice2022", "Jio2022", "STC2022", "TurkTelecom2022Leak", "BSNL2023",
        "Rostelecom2023Leak", "OrangeSpain2023", "Telia2023",
    ]

    outage_bases = [
        "Facebook", "Rogers_Canada", "KDDI_Japan", "Spark_NZ", "Optus_Aus",
        "Pakistan2011", "HurricaneSandy2012", "Syria2013", "Turkey2014",
        "Libya2014", "BT2015", "Telia2016", "DYN2016", "Comcast2017",
        "Iran2019", "Cloudflare2020", "Google2020", "Fastly2021", "Akamai2021",
        "AWS2021", "Lumen2022", "Slack2022", "Cloudflare2022", "Azure2022",
        "Iran2022", "Telstra2023", "OrangeSpain2023Out", "VodafoneUK2023",
        "Swisscom2023", "NTT2023",
    ]

    if "Hijack" in event_name or any(x in base for x in hijack_bases):
        return "Origin/Path Hijack"
    if "Leak" in event_name or any(x in base for x in leak_bases):
        return "Route Leak"
    if "Outage" in event_name or any(x in base for x in outage_bases):
        return "Outage"
    return "Other"

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

def get_models():
    return {
        "Logistic_Reg":  LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10,
                                                random_state=42, n_jobs=-1),
        "XGBoost":       xgb.XGBClassifier(eval_metric='logloss',
                                            random_state=42, n_jobs=-1),
        "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                       early_stopping=True, random_state=42),
    }

# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess(df):
    """
    Add metadata columns and apply event-isolated Z-score normalization.

    The scaler is fitted exclusively on normal (baseline) windows of each
    incident and applied to all windows of that incident. This removes the
    Year Effect without leaking information across incidents.
    """
    df["Incidente_Base"] = df["Evento"].apply(
        lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x))
    df["Categoria"] = df["Evento"].apply(get_category)
    df = df[df["Categoria"] != "Other"].copy()

    # Feature columns: exclude all metadata
    cols_X = df.drop(columns=[
        "Timestamp", "Evento", "Incidente_Base", "Label",
        "Colector", "Categoria"
    ]).columns.tolist()

    # Event-isolated Z-score normalization
    for base in df["Incidente_Base"].unique():
        idx_all    = df["Incidente_Base"] == base
        idx_normal = idx_all & (df["Label"] == 0)
        if idx_normal.sum() == 0:
            continue
        scaler = StandardScaler()
        scaler.fit(df.loc[idx_normal, cols_X])
        df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])

    return df, cols_X

# ==============================================================================
# LOEO CROSS-VALIDATION
# ==============================================================================

def run_loeo(df, cols_X, model_name, model, csv_file):
    """
    Run one full LOEO-CV loop for a single model and return a results dict.
    """
    X          = df[cols_X].values
    y          = df["Label"].values
    incidents  = df["Incidente_Base"].values
    categories = df["Categoria"].values

    unique_incidents = np.unique(incidents)

    y_true_all, y_pred_all, cats_all = [], [], []

    for incident_test in unique_incidents:
        mask_test  = incidents == incident_test
        mask_train = ~mask_test

        X_train, y_train = X[mask_train], y[mask_train]
        X_test,  y_test  = X[mask_test],  y[mask_test]
        cats_test = categories[mask_test]

        # Skip if test set contains only one class
        if len(np.unique(y_test)) < 2:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        cats_all.extend(cats_test)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    cats_all   = np.array(cats_all)

    result = {
        "Dataset":    csv_file,
        "Modelo":     model_name,
        "Acc_Global": round(accuracy_score(y_true_all, y_pred_all), 6),
        "F1_Global":  round(f1_score(y_true_all, y_pred_all), 6),
        "Rec_Global": round(recall_score(y_true_all, y_pred_all), 6),
    }

    for cat in ["Origin/Path Hijack", "Route Leak", "Outage"]:
        mask = cats_all == cat
        y_t, y_p = y_true_all[mask], y_pred_all[mask]
        result[f"Rec_{cat}"] = round(
            recall_score(y_t, y_p) if len(y_t) > 0 and len(np.unique(y_t)) > 1 else 0.0, 6)

    return result

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    all_results = []

    for csv_file in CSV_FILES:
        print(f"\n{'='*70}")
        print(f"Processing: {csv_file}")
        print(f"{'='*70}")

        df = pd.read_csv(os.path.join(CSV_FOLDER, csv_file))
        df, cols_X = preprocess(df)

        unique_incidents = df["Incidente_Base"].nunique()
        print(f"Unique incidents for LOEO-CV: {unique_incidents}")

        models = get_models()
        for model_name, model in models.items():
            print(f"  Training {model_name}...", end=" ", flush=True)
            result = run_loeo(df, cols_X, model_name, model, csv_file)
            all_results.append(result)
            print(f"Acc={result['Acc_Global']:.4f}  "
                  f"F1={result['F1_Global']:.4f}  "
                  f"Rec={result['Rec_Global']:.4f}")

    # --- Final report ---
    df_results = pd.DataFrame(all_results)

    print("\n" + "="*110)
    print(" FINAL RESULTS — LEAVE-ONE-EVENT-OUT CV")
    print("="*110)
    print(df_results.to_string(index=False))

    df_results.to_csv("resultados_LOEO_final.csv", index=False)
    print("\nResults saved to: resultados_LOEO_final.csv")