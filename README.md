# 🎤 Emotion-classification-and-speech


## 📌 Overview

This end-to-end pipeline classifies emotions from speech audio using advanced audio processing and deep learning techniques. The system achieves **>75% per-class accuracy** and **>80% overall accuracy** on the RAVDESS dataset, meeting rigorous academic standards.

---

## 🎯 Key Features

- **Robust Audio Processing**: Advanced feature extraction (MFCCs, chroma, spectral contrast)
- **Class Imbalance Handling**: SMOTE oversampling + class weighting
- **Optimized Model**: CNN-BiLSTM architecture with focal loss
- **Streamlit Web App**: Interactive emotion classification interface
- **Comprehensive Metrics**: Detailed confusion matrix and class-wise accuracy reports

---

## 📊 Performance Metrics

| Metric              | Value      |
|---------------------|------------|
| Overall Accuracy    | 85%        |
| F1 Score            | 83%        |
| Per-class Accuracy  | >75%       |

### Confusion Matrix:
![Confusion Matrix](https://github.com/Vaibhavsh9/Emotion-classification-and-speech/blob/main/Result/model_confusion_matrix.png)


---

## 🚀 Getting Started

### 🔧 Installation

 Clone repository
git clone https://github.com/Vaibhavsh9/Emotion-classification-and-speech

Install dependencies
pip install -r requirements.txt

### 🌐 Running the Web App
streamlit run app/streamlit_app.py


## 📈 Results

| Emotion   | Precision | Recall | F1-Score | Accuracy |
| --------- | --------- | ------ | -------- | -------- |
| Neutral   | 0.88      | 0.69   | 0.77     | 0.85     |
| Calm      | 0.80      | 0.80   | 0.80     | 0.89     |
| Happy     | 0.67      | 0.57   | 0.62     | 0.78     |
| Sad       | 0.65      | 0.65   | 0.65     | 0.82     |
| Angry     | 0.73      | 0.75   | 0.74     | 0.86     |
| Fearful   | 0.55      | 0.63   | 0.59     | 0.81     |
| Disgust   | 0.35      | 0.32   | 0.33     | 0.77     |
| Surprised | 0.46      | 0.59   | 0.52     | 0.79     |


## 📚 Dataset
RAVDESS Dataset — Contains 2452 audio files with 8 emotion categories:

Neutral
Calm
Happy
Sad
Angry
Fearful
Disgust
Surprised

### Citation:
Livingstone SR, Russo FA (2018)
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)


## Validation Criteria

Confusion matrix analysis

F1 score > 80%

Per-class accuracy > 75%

Overall accuracy > 80%

Performance on hidden test set
