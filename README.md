# ECG Physical Activity Analysis

This repository contains a clean Python implementation accompanying the paper **“Statistical Analysis of Physical Activity’s Impact on Cardiovascular Health Using ECG and Lifestyle Metrics.”**

The code includes ECG filtering, ECG/HRV feature extraction, statistical comparison of activity groups, K-means clustering, and MVPA-based logistic regression.

## Use

```bash
pip install -r requirements.txt
python run_analysis.py --ecg-dir path/to/ecg_files --lifestyle path/to/lifestyle.csv --output results
```

ECG files should be Excel files with time in the first column and ECG amplitude in the second column. The lifestyle file must contain `subject_id` and `mvpa_minutes`; `subject_id` should match the ECG filename without its extension.

The participant dataset is not included.
