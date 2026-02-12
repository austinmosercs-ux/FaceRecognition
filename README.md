# FaceRecognition

## Feature Extraction Report
This project reads the AR face database 22-point markup files and extracts seven feature ratios.
It produces a single consolidated text report.

### Run
```zsh
python3 face_feature_extraction.py
```

### Output
- `facial_features_report.txt`: consolidated report with detailed per-file features, per-person averages,
  gender comparison, summary statistics, and key insights.

### Notes
- The database path is currently set in `face_feature_extraction.py` under `__main__`.
