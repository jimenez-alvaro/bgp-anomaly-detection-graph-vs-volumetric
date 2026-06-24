# BGP Anomaly Detection: Graph Topology vs. Volumetric Features

Code and dataset pipeline for the paper:

> **Graph Topology vs. Volumetric Features for BGP Anomaly Detection: A Large-Scale Empirical Evaluation**  
> Álvaro Jiménez Martín, Shadi Motaali, Jorge E. López de Vergara Méndez  
> Escuela Politécnica Superior, Universidad Autónoma de Madrid  
> Submitted to CNSM 2026

---

## Overview

This repository contains the full reproducible pipeline for comparing volumetric and graph-topological feature paradigms for supervised BGP anomaly detection. The dataset covers 120 real-world BGP incidents between 2004 and 2023, spanning four anomaly categories, observed from RIPE RIS collector rrc04 (Geneva, Europe).

Key contributions:
- Large-scale controlled comparison: 120 incidents, 4 anomaly categories, 19-year span
- Leave-One-Event-Out (LOEO) cross-validation to prevent information leakage between incidents
- Event-isolated Z-score normalization to mitigate the Year Effect
- SHAP-based explainability analysis

---

## Results Summary

| Paradigm   | Model    | Accuracy | F1    | Recall |
|------------|----------|----------|-------|--------|
| Volumetric | LR       | 0.599    | 0.524 | 0.439  |
| Volumetric | RF       | 0.693    | 0.615 | 0.488  |
| Volumetric | XGBoost  | 0.716    | 0.671 | 0.577  |
| Volumetric | MLP      | 0.702    | 0.665 | 0.588  |
| **Topological** | LR  | 0.863    | 0.843 | 0.730  |
| **Topological** | RF  | 0.969    | 0.968 | 0.944  |
| **Topological** | XGBoost | 0.966 | 0.965 | 0.938 |
| **Topological** | MLP | 0.969    | 0.969 | 0.942  |

Evaluated under strict LOEO cross-validation on rrc04.

---

## Repository Structure

```
bgp-anomaly-detection-graph-vs-volumetric/
│
├── 1_Extraction.py   # Download raw BGP data and extract features via BML
├── 2_Json_to_csv.py          # Consolidate BML JSON outputs into CSV files
├── 3_LOEO_training.py        # LOEO-CV training and evaluation (4 models x 2 paradigms)
├── 4_SHAP_analysis.py        # SHAP explainability analysis for XGBoost
├── 5_Confusion_matrices.py   # Generate cumulative LOEO confusion matrix figures
│
└── README.md
```
---

## Requirements

Python 3.8+ and the following packages:

```
bml
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

BML must be installed separately following the [official BML documentation](https://github.com/BMLResearch/BML).

---

## Reproducibility

### Step 1 — Extract features

Downloads raw BGP data from RIPE RIS and extracts volumetric and topological features for all 120 incidents:

```bash
python bgp_dataset_extraction.py
```

Output: `dataset_grande/` with four category subfolders, each containing JSON feature files.

> **Note:** This step requires a stable internet connection and significant disk space (~10–20 GB). Processing time depends on available CPU and network speed.

### Step 2 — Consolidate to CSV

```bash
python bgp_json_to_csv.py
```

Output:
- `volumen_rrc04_final.csv` — 32 volumetric features
- `grafos_rrc04_final.csv`  — 14 topological features

Move both files to `dataset_csv_grande/` before running the next steps.

### Step 3 — Train models (LOEO-CV)

```bash
python bgp_loeo_training.py
```

Output: `resultados_LOEO_final.csv` with global and per-category metrics for all models.

### Step 4 — SHAP analysis

```bash
python bgp_shap_analysis.py
```

Output: bar and dot plots in `shap_outputs/`.

### Step 5 — Confusion matrices

```bash
python bgp_confusion_matrices.py
```

Output: PDF confusion matrices in `confusion_matrix_outputs/`.

---

## Dataset

Incidents are sourced from publicly available BGP event databases including:
- [RIPE NCC Routing Information Service (RIS)](https://ris.ripe.net/)
- [BGPStream](https://bgpstream.caida.org/)
- [Cloudflare Radar](https://radar.cloudflare.com/)

All raw data is downloaded directly from RIPE RIS archives via the BML framework during Step 1.

---

## Citation

If you use this code or dataset pipeline in your research, please cite:

```bibtex
@inproceedings{jimenez2026bgp,
  title     = {Graph Topology vs. Volumetric Features for {BGP} Anomaly Detection:
               A Large-Scale Empirical Evaluation},
  author    = {Jim\'enez Mart\'in, \'Alvaro and Motaali, Shadi and
               L\'opez de Vergara M\'endez, Jorge E.},
  booktitle = {Proc. 22nd Int. Conf. Network and Service Management (CNSM)},
  year      = {2026},
  address   = {Alcal\'a de Henares, Spain}
}
```

---

## License

This project is released under the MIT License.
