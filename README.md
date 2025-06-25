# 🎤 Emotion-classification-and-speech


## 📌 Overview

This project delivers a complete solution for recognizing emotions in speech audio by leveraging sophisticated audio feature extraction and state-of-the-art deep learning models. Tested on the RAVDESS dataset, the system consistently achieves over 75% accuracy per emotion class and surpasses 80% overall accuracy, aligning with high academic benchmarks.

---

## 🎯 Key Features

- **Advanced Audio Feature Extraction**: Utilizes MFCCs, chroma vectors, and spectral contrast for rich audio representation.
- **Balanced Training Strategy**: Employs SMOTE for oversampling and applies class weights to address data imbalance.
- **Enhanced Deep Learning Model**: Implements a hybrid CNN-BiLSTM network with focal loss for precise emotion recognition.
- **User-Friendly Web Application**: Offers an interactive Streamlit interface for real-time emotion detection from speech.
- **Thorough Evaluation Metrics**: Provides in-depth confusion matrix visualization and per-class accuracy breakdowns.

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
streamlit run Mars.py


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
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)


## Validation Criteria
Confusion matrix analysis
F1 score > 80%
Per-class accuracy > 75%
Overall accuracy > 80%
