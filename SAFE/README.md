# SAFE Dataset Replication - Machine Learning on Spectrogram Features

This repository contains code to replicate the results from Section 4.2 and 5.2 of the SAFE (Sound Analysis for Fall Events) paper:
**"SAFE: Sound Analysis for Fall Event detection using machine learning"** by Garcia & Huang (2025).

## Overview

This implementation focuses on:
- **Section 4.2**: Machine Learning algorithms applied to spectrogram features
- **Section 5.2**: Evaluation results for ML models on spectrogram-based features

## Dataset

The SAFE dataset contains 950 audio samples:
- 475 fall events
- 475 non-fall events

Dataset link: https://www.kaggle.com/dsv/7516242

## Features

This replication implements **Section 5.2 Table 4** spectrogram feature extraction methods:

1. **Mel Spectrogram**
2. **STFT Spectrogram** (Short-Time Fourier Transform)
3. **MFCC Spectrogram** (Mel-Frequency Cepstral Coefficients)
4. **Constant Q Transform (CQT) Spectrogram**
5. **Continuous Wavelet Transform (CWT) Spectrogram**
6. **Chroma Features Spectrogram**

2. **Machine Learning Models**:
   - Logistic Regression (Linear)
   - Linear SVM
   - Decision Tree
   - Random Forest
   - Extra Trees Classifier
   - Gradient Boosting

3. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place the SAFE dataset in the `../data/` folder (or update path in `data_loader.py`)

2. Run the main script:
```bash
python main.py
```

This will:
- Load and preprocess the SAFE dataset
- Extract spectrogram features
- Train ML models
- Evaluate and report results

## Expected Results

According to the SAFE paper (Section 5.2):
- Linear models achieve up to **97% accuracy** on spectrogram features
- ML models show high performance indicating separability of audio features
- Results emphasize the viability of sound-based fall detection

## Reference

Garcia, A., & Huang, X. (2025). SAFE: Sound Analysis for Fall Event detection using machine learning. *Smart Health*, 35, 100539. https://doi.org/10.1016/j.smhl.2024.100539



## NAstepne kroki Next steps: 
 - zrob sobie CURSOS PRO. 


 - zreplikuj tabele 3 i tabele 5 i bedziesz mial zreplikowany paper SAFE
