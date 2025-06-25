import sys
import os
import librosa
import numpy as np
import joblib
from multipart import file_path
from tensorflow.keras.models import load_model


# Focal loss implementation for model loading
def focal_loss(gamma=2.0, alpha=0.75):
    """Focal loss implementation required for model loading"""

    def focal_loss_fn(y_true, y_pred):
        import tensorflow.keras.backend as K
        from tensorflow.keras.losses import sparse_categorical_crossentropy
        ce = sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = K.exp(-ce)
        loss = alpha * K.pow(1 - p_t, gamma) * ce
        return K.mean(loss)

    return focal_loss_fn


def load_and_process_audio(file_path, sr=22050, duration=3):
    """Load and process audio file with error handling"""
    file_path=r"/Users/vs/Downloads/Audio_Speech_Actors_01-24/Actor_05/03-01-02-01-01-02-05.wav"
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        if len(audio) > sr * duration:
            start = (len(audio) - sr * duration) // 2
            audio = audio[start:start + sr * duration]
        else:
            audio = np.pad(audio, (0, max(0, sr * duration - len(audio))), 'constant')
        return audio
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None


def extract_features(audio, sr=22050):
    """Extract audio features for prediction"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1),
        np.mean(contrast, axis=1),
        librosa.feature.rms(y=audio).mean(),
        librosa.feature.zero_crossing_rate(audio).mean()
    ])


def load_emotion_labels():
    """Load emotion labels with fallback to default"""
    try:
        import json
        if os.path.exists('emotion_categories.json'):
            with open('emotion_categories.json', 'r') as f:
                return json.load(f)
    except:
        pass
    return ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def main():
    # Load model and scaler with custom objects
    try:
        model = load_model('speech_emotion_model.h5', custom_objects={'focal_loss': focal_loss()})
        scaler = joblib.load('feature_scaler.pkl')
        if not hasattr(scaler, 'transform'):
            raise TypeError("Loaded scaler is not a valid Scikit-learn scaler object.")

    except Exception as e:
        print(f"Error loading model files: {e}")
        print("Ensure 'emotion_model.h5' and 'scaler.pkl' are in the current directory")
        sys.exit(1)

    # Load emotion labels
    emotion_labels = load_emotion_labels()

    # Get audio path - support both command-line and direct specification
    audio_path = None

    # Option 1: Check command-line argument
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]

    # Option 2: Use hardcoded path if not provided
    if not audio_path:
        audio_path = "/Users/vs/Downloads/Audio_Speech_Actors_01-24/Actor_05/03-01-02-01-01-02-05.wav"
        print(f"Using default audio file: {audio_path}")

    # Verify file existence
    if not os.path.exists(audio_path):
        print(f"Error: File not found - {audio_path}")
        sys.exit(1)

    # Process audio
    audio = load_and_process_audio(audio_path)
    if audio is None:
        sys.exit(1)

    # Extract features
    features = extract_features(audio)

    # Scale features
    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        print(f"Error scaling features: {e}")
        sys.exit(1)

    # Reshape for model input
    features_reshaped = features_scaled[..., np.newaxis]

    # Predict emotion
    try:
        emotion_prob = model.predict(features_reshaped, verbose=0)
        emotion_idx = np.argmax(emotion_prob)
        confidence = emotion_prob[0][emotion_idx]

        # Handle filtered classes
        if emotion_idx < len(emotion_labels):
            emotion_name = emotion_labels[emotion_idx]
        else:
            emotion_name = "unknown"

        print(f"Predicted Emotion: {emotion_name}")
        print(f"Confidence: {confidence:.4f}")
        print("All Probabilities:")
        for i, prob in enumerate(emotion_prob[0]):
            label = emotion_labels[i] if i < len(emotion_labels) else f"Class_{i}"
            print(f"{label}: {prob:.4f}")
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
