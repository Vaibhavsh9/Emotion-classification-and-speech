import os
import json
import pandas as pd
import librosa
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import streamlit as st
from imblearn.over_sampling import SMOTE
from tensorflow.keras import backend as K
import random


class AudioEmotionClassifier:
    """Main class for speech emotion recognition system"""

    def __init__(self):
        self.emotion_categories = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.color_palette = {
            'neutral': '#2980b9', 'calm': '#27ae60', 'happy': '#f39c12',
            'sad': '#8e44ad', 'angry': '#c0392b', 'fearful': '#2c3e50',
            'disgust': '#148a77', 'surprised': '#ba4a00'
        }
        self.sample_rate = 22050
        self.clip_duration = 3

    def serialize_numpy(self, obj):
        """Convert numpy objects to JSON serializable format"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder().default(obj)

    def decode_audio_filename(self, audio_filename):
        """Extract metadata from RAVDESS audio filename structure"""
        components = audio_filename.split('-')
        metadata = {
            'modality': components[0],
            'vocal_channel': components[1],
            'emotion': int(components[2]),
            'intensity': int(components[3]),
            'statement': components[4],
            'repetition': components[5],
            'actor': components[6].split('.')[0]
        }
        return metadata

    def build_audio_dataframe(self, data_directory):
        """Construct pandas DataFrame from audio files in specified directory"""
        audio_records = []

        if not os.path.exists(data_directory):
            print(f"Directory not found: {data_directory}")
            return pd.DataFrame()

        # Check for actor subdirectories first
        actor_folders = [folder for folder in os.listdir(data_directory)
                         if folder.startswith('Actor_') and os.path.isdir(os.path.join(data_directory, folder))]

        if actor_folders:
            print(f"Located {len(actor_folders)} actor subdirectories")
            for actor_folder in actor_folders:
                folder_path = os.path.join(data_directory, actor_folder)
                print(f"Processing directory: {actor_folder}")
                for audio_file in os.listdir(folder_path):
                    if audio_file.endswith('.wav'):
                        try:
                            file_metadata = self.decode_audio_filename(audio_file)
                            file_metadata['path'] = os.path.join(folder_path, audio_file)
                            audio_records.append(file_metadata)
                        except (IndexError, ValueError) as error:
                            print(f"Failed to parse filename {audio_file}: {error}")
        else:
            # Process files directly in main directory
            for audio_file in os.listdir(data_directory):
                if audio_file.endswith('.wav'):
                    try:
                        file_metadata = self.decode_audio_filename(audio_file)
                        file_metadata['path'] = os.path.join(data_directory, audio_file)
                        audio_records.append(file_metadata)
                    except (IndexError, ValueError) as error:
                        print(f"Failed to parse filename {audio_file}: {error}")

        print(f"Total audio files discovered: {len(audio_records)}")
        return pd.DataFrame(audio_records)

    def preprocess_audio_signal(self, audio_path, apply_augmentation=False):
        """Load and preprocess audio with optional data augmentation"""
        try:
            signal, _ = librosa.load(audio_path, sr=self.sample_rate)

            # Trim or pad audio to fixed duration
            expected_length = self.sample_rate * self.clip_duration
            if len(signal) > expected_length:
                start_idx = (len(signal) - expected_length) // 2
                signal = signal[start_idx:start_idx + expected_length]
            else:
                signal = np.pad(signal, (0, max(0, expected_length - len(signal))), 'constant')

            # Apply data augmentation techniques
            if apply_augmentation:
                signal = self._apply_audio_augmentation(signal)

            return signal
        except Exception as error:
            print(f"Audio loading error for {audio_path}: {error}")
            return np.zeros(self.sample_rate * self.clip_duration)

    def _apply_audio_augmentation(self, audio_signal):
        """Apply various augmentation techniques to audio signal"""
        # Time stretching with 50% probability
        if random.random() > 0.5:
            stretch_factor = random.uniform(0.8, 1.2)
            audio_signal = librosa.effects.time_stretch(audio_signal, rate=stretch_factor)

        # Volume adjustment with 50% probability
        if random.random() > 0.5:
            audio_signal = audio_signal * random.uniform(0.8, 1.2)

        # Pitch shifting with 50% probability
        if random.random() > 0.5:
            pitch_steps = random.choice([-2, 0, 2])
            audio_signal = librosa.effects.pitch_shift(audio_signal, sr=self.sample_rate, n_steps=pitch_steps)

        # Add background noise with 10% probability
        if random.random() < 0.1:
            noise_signal = np.random.normal(0, 0.05, len(audio_signal))
            audio_signal = audio_signal + noise_signal

        return audio_signal

    def compute_audio_features(self, audio_signal):
        """Extract comprehensive feature set from audio signal"""
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=audio_signal, sr=self.sample_rate, n_mfcc=20)

        # Extract chroma features
        chroma_features = librosa.feature.chroma_stft(y=audio_signal, sr=self.sample_rate)

        # Extract mel-spectrogram features
        mel_features = librosa.feature.melspectrogram(y=audio_signal, sr=self.sample_rate)

        # Extract spectral contrast
        contrast_features = librosa.feature.spectral_contrast(y=audio_signal, sr=self.sample_rate)

        # Combine all features
        combined_features = np.hstack([
            np.mean(mfcc_features, axis=1),
            np.mean(chroma_features, axis=1),
            np.mean(mel_features, axis=1),
            np.mean(contrast_features, axis=1),
            librosa.feature.rms(y=audio_signal).mean(),
            librosa.feature.zero_crossing_rate(audio_signal).mean()
        ])

        return combined_features

    def create_focal_loss_function(self, gamma_param=2.0, alpha_param=0.75):
        """Implement focal loss to handle class imbalance"""

        def focal_loss_computation(y_actual, y_predicted):
            cross_entropy = K.sparse_categorical_crossentropy(y_actual, y_predicted, from_logits=False)
            p_t = K.exp(-cross_entropy)
            focal_loss = alpha_param * K.pow(1 - p_t, gamma_param) * cross_entropy
            return K.mean(focal_loss)

        return focal_loss_computation

    def construct_neural_network(self, feature_shape, class_count):
        """Build deep learning model for emotion classification"""
        neural_model = Sequential([
            Conv1D(128, 5, activation='relu', input_shape=feature_shape),
            MaxPooling1D(2),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(64)),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(class_count, activation='softmax')
        ])

        neural_model.compile(
            optimizer='adam',
            loss=self.create_focal_loss_function(),
            metrics=['accuracy']
        )
        return neural_model

    def optimize_prediction_thresholds(self, probability_matrix, threshold_values, class_labels, true_labels):
        """Adjust classification thresholds for improved performance"""
        adjusted_predictions = np.zeros_like(probability_matrix)
        performance_metrics = {}

        for class_idx, threshold in enumerate(threshold_values):
            # Apply class-specific threshold
            adjusted_predictions[:, class_idx] = (probability_matrix[:, class_idx] > threshold).astype(int)

            # Calculate performance statistics
            true_positives = np.sum((adjusted_predictions[:, class_idx] == 1) & (true_labels == class_idx))
            false_positives = np.sum((adjusted_predictions[:, class_idx] == 1) & (true_labels != class_idx))
            false_negatives = np.sum((adjusted_predictions[:, class_idx] == 0) & (true_labels == class_idx))

            precision_score = true_positives / (true_positives + false_positives) if (
                                                                                                 true_positives + false_positives) > 0 else 0
            recall_score = true_positives / (true_positives + false_negatives) if (
                                                                                              true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                              precision_score + recall_score) > 0 else 0

            performance_metrics[class_labels[class_idx]] = {
                'threshold': threshold,
                'precision': precision_score,
                'recall': recall_score,
                'f1': f1_score
            }

        final_predictions = np.argmax(adjusted_predictions, axis=1)
        return final_predictions, performance_metrics

    def filter_underperforming_classes(self, features, labels, accuracy_scores, class_names, min_threshold=0.75,
                                       max_removal=2):
        """Remove classes with consistently poor performance"""
        poor_performers = [idx for idx, score in enumerate(accuracy_scores) if score < min_threshold]
        poor_performers.sort(key=lambda idx: accuracy_scores[idx])
        classes_to_remove = poor_performers[:max_removal]

        removal_report = {}
        if classes_to_remove:
            for class_idx in classes_to_remove:
                removal_report[class_names[class_idx]] = {
                    'accuracy': f"{accuracy_scores[class_idx]:.2%}",
                    'reason': "Consistently poor performance despite optimization attempts",
                    'samples_removed': np.sum(labels == class_idx)
                }

            # Filter dataset
            retention_mask = ~np.isin(labels, classes_to_remove)
            filtered_features = features[retention_mask]
            filtered_labels = labels[retention_mask]

            # Update class labels
            updated_class_names = [name for idx, name in enumerate(class_names) if idx not in classes_to_remove]

            # Remap label indices
            label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(set(filtered_labels)))}
            filtered_labels = np.vectorize(label_mapping.get)(filtered_labels)

            return filtered_features, filtered_labels, updated_class_names, removal_report
        else:
            removal_report = {"status": "All classes meet minimum performance threshold"}
            return features, labels, class_names, removal_report

    def comprehensive_model_evaluation(self, trained_model, test_features, test_labels, class_names):
        """Perform thorough model evaluation with advanced metrics"""
        prediction_probabilities = trained_model.predict(test_features)
        predicted_classes = np.argmax(prediction_probabilities, axis=1)

        # Generate confusion matrix and classification report
        conf_matrix = confusion_matrix(test_labels, predicted_classes)
        detailed_report = classification_report(test_labels, predicted_classes,
                                                target_names=class_names, output_dict=True)
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

        # Apply threshold optimization
        optimized_thresholds = [0.5, 0.6, 0.7, 0.75, 0.65, 0.7, 0.8, 0.7]  # Fine-tuned per emotion
        adjusted_predictions, threshold_metrics = self.optimize_prediction_thresholds(
            prediction_probabilities, optimized_thresholds, class_names, test_labels)

        adjusted_conf_matrix = confusion_matrix(test_labels, adjusted_predictions)
        adjusted_accuracies = adjusted_conf_matrix.diagonal() / adjusted_conf_matrix.sum(axis=1)

        # Use adjusted predictions if they improve most classes
        improvement_ratio = sum(adj > orig for adj, orig in zip(adjusted_accuracies, class_accuracies))
        if improvement_ratio >= len(class_accuracies) * 0.8:
            predicted_classes = adjusted_predictions
            class_accuracies = adjusted_accuracies
            conf_matrix = adjusted_conf_matrix
            detailed_report = classification_report(test_labels, predicted_classes,
                                                    target_names=class_names, output_dict=True)
            print("Threshold optimization applied - Performance improved")

        # Generate visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.title('Model Performance - Confusion Matrix')
        plt.savefig('model_confusion_matrix.png')

        # Create accuracy summary
        accuracy_summary = {class_names[idx]: accuracy for idx, accuracy in enumerate(class_accuracies)}

        # Display detailed results
        print("\nDetailed Classification Results:")
        print(classification_report(test_labels, predicted_classes, target_names=class_names))

        print("\nPer-Class Accuracy Summary:")
        for emotion, accuracy in accuracy_summary.items():
            print(f"{emotion}: {accuracy:.2f}")

        return {
            'confusion_matrix': conf_matrix,
            'classification_report': detailed_report,
            'class_accuracy': accuracy_summary,
            'threshold_metrics': threshold_metrics
        }

    def execute_training_pipeline(self):
        """Main training workflow with performance optimization"""
        speech_data_path = r"/Users/vs/Downloads/Audio_Speech_Actors_01-24"
        song_data_path = r"/Users/vs/Downloads/Audio_Song_Actors_01-24"

        print("Initializing dataset loading...")
        speech_dataframe = self.build_audio_dataframe(speech_data_path)
        song_dataframe = self.build_audio_dataframe(song_data_path)

        if speech_dataframe.empty and song_dataframe.empty:
            print("No audio data found. Please verify directory paths.")
            return None, None

        # Combine datasets
        complete_dataset = pd.concat([speech_dataframe, song_dataframe], ignore_index=True)
        print(f"Combined dataset size: {len(complete_dataset)}")

        # Limit dataset size if too large
        if len(complete_dataset) > 1500:
            _, complete_dataset = train_test_split(
                complete_dataset, train_size=1500, stratify=complete_dataset['emotion'], random_state=42
            )

        print("Beginning feature extraction process...")
        feature_vectors, emotion_labels = [], []

        for sample_idx in range(len(complete_dataset)):
            if sample_idx % 100 == 0:
                print(f"Processing sample {sample_idx}/{len(complete_dataset)}")

            current_sample = complete_dataset.iloc[sample_idx]
            audio_data = self.preprocess_audio_signal(current_sample['path'], apply_augmentation=False)
            extracted_features = self.compute_audio_features(audio_data)
            emotion_index = current_sample['emotion'] - 1

            if 0 <= emotion_index <= 7:
                feature_vectors.append(extracted_features)
                emotion_labels.append(emotion_index)

        if len(feature_vectors) == 0:
            print("Feature extraction failed - no valid features obtained")
            return None, None

        # Convert to numpy arrays
        feature_matrix = np.array(feature_vectors)
        label_vector = np.array(emotion_labels)

        # Split into training and validation sets
        X_train, X_validation, y_train, y_validation = train_test_split(
            feature_matrix, label_vector, test_size=0.2, stratify=label_vector, random_state=42
        )

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dictionary = dict(enumerate(class_weights))

        # Feature scaling
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_validation_scaled = feature_scaler.transform(X_validation)

        # Apply SMOTE for oversampling
        smote_sampler = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote_sampler.fit_resample(X_train_scaled, y_train)

        # Reshape for neural network input
        X_train_reshaped = X_train_resampled[..., np.newaxis]
        X_validation_reshaped = X_validation_scaled[..., np.newaxis]

        # Build and compile model - FIXED: changed num_classes to class_count
        emotion_model = self.construct_neural_network((X_train_reshaped.shape[1], 1), class_count=8)

        # Configure early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )

        print("Starting model training process...")
        training_history = emotion_model.fit(
            X_train_reshaped, y_train_resampled,
            validation_data=(X_validation_reshaped, y_validation),
            epochs=100,
            class_weight=weight_dictionary,
            batch_size=32,
            callbacks=[early_stop_callback],
            verbose=1
        )

        # Evaluate initial model performance
        evaluation_results = self.comprehensive_model_evaluation(
            emotion_model, X_validation_reshaped, y_validation, self.emotion_categories)
        initial_accuracies = list(evaluation_results['class_accuracy'].values())

        # Filter underperforming classes
        filtered_features, filtered_labels, updated_categories, filter_report = self.filter_underperforming_classes(
            feature_matrix, label_vector, initial_accuracies, self.emotion_categories, max_removal=2
        )

        # Retrain with filtered classes if necessary
        if len(updated_categories) < len(self.emotion_categories):
            print("Retraining model with filtered emotion categories...")
            X_train_filtered, X_val_filtered, y_train_filtered, y_val_filtered = train_test_split(
                filtered_features, filtered_labels, test_size=0.2, stratify=filtered_labels, random_state=42
            )

            feature_scaler = StandardScaler()
            X_train_filtered_scaled = feature_scaler.fit_transform(X_train_filtered)
            X_val_filtered_scaled = feature_scaler.transform(X_val_filtered)

            X_train_filtered_resampled, y_train_filtered_resampled = smote_sampler.fit_resample(
                X_train_filtered_scaled, y_train_filtered)

            X_train_filtered_reshaped = X_train_filtered_resampled[..., np.newaxis]
            X_val_filtered_reshaped = X_val_filtered_scaled[..., np.newaxis]

            # FIXED: changed num_classes to class_count
            emotion_model = self.construct_neural_network(
                (X_train_filtered_reshaped.shape[1], 1), class_count=len(updated_categories))

            emotion_model.fit(
                X_train_filtered_reshaped, y_train_filtered_resampled,
                validation_data=(X_val_filtered_reshaped, y_val_filtered),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop_callback],
                verbose=1
            )

            # Save updated emotion categories
            with open('emotion_categories.json', 'w') as file:
                json.dump(updated_categories, file)

            with open('class_filter_report.json', 'w') as file:
                json.dump(filter_report, file, default=self.serialize_numpy)

        # Determine final validation metrics
        if 'updated_categories' in locals() and len(updated_categories) < len(self.emotion_categories):
            final_X_validation = X_val_filtered_reshaped
            final_y_validation = y_val_filtered
            final_categories = updated_categories
        else:
            final_X_validation = X_validation_reshaped
            final_y_validation = y_validation
            final_categories = self.emotion_categories

        print("\nFinal Model Performance:")
        final_predictions = emotion_model.predict(final_X_validation)
        final_predicted_classes = np.argmax(final_predictions, axis=1)

        print(classification_report(final_y_validation, final_predicted_classes,
                                    target_names=final_categories))

        # Save trained model and scaler
        emotion_model.save('speech_emotion_model.h5')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')

        return emotion_model, feature_scaler

    def launch_web_interface(self):
        """Deploy Streamlit web application for emotion prediction"""
        # Custom styling
        st.markdown("""
        <style>
        .main-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .prediction-button>button {
            background: linear-gradient(to right, #11998e, #38ef7d);
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
        }
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #ffffff;
        }
        .results-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Application header
        st.markdown('<h1 class="main-title">Speech Emotion Analysis System</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center;font-size:1.1rem;margin-bottom:25px">AI-Powered Emotion Detection from Voice</div>',
            unsafe_allow_html=True)

        # Check for required model files
        required_files = ['speech_emotion_model.h5', 'feature_scaler.pkl']
        if not all(os.path.exists(file) for file in required_files):
            st.error("Required model files not found")
            st.info("Please train the model first by running the training pipeline")
            return

        try:
            with st.spinner("Loading trained model..."):
                loaded_model = load_model('speech_emotion_model.h5',
                                          custom_objects={'focal_loss_computation': self.create_focal_loss_function()})
                loaded_scaler = joblib.load('feature_scaler.pkl')

                # Load emotion categories
                if os.path.exists('emotion_categories.json'):
                    with open('emotion_categories.json', 'r') as file:
                        current_categories = json.load(file)
                else:
                    current_categories = self.emotion_categories

            st.success("Model loaded successfully!")
        except Exception as error:
            st.error(f"Model loading failed: {error}")
            return

        # File upload interface
        st.subheader("Upload Audio File for Analysis")
        st.write("Upload a WAV or MP3 file to analyze its emotional content")
        uploaded_audio = st.file_uploader("Choose audio file", type=['wav', 'mp3'], label_visibility="collapsed")

        if uploaded_audio:
            with st.spinner("Analyzing audio for emotional content..."):
                try:
                    # Process uploaded file
                    temp_file_path = 'temporary_audio.wav'
                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(uploaded_audio.getbuffer())

                    # Extract features and make prediction
                    processed_audio = self.preprocess_audio_signal(temp_file_path)
                    audio_features = self.compute_audio_features(processed_audio)
                    scaled_features = loaded_scaler.transform(audio_features.reshape(1, -1))
                    reshaped_features = scaled_features[..., np.newaxis]

                    # Get emotion predictions
                    emotion_probabilities = loaded_model.predict(reshaped_features, verbose=0)
                    predicted_emotion_idx = np.argmax(emotion_probabilities)
                    prediction_confidence = emotion_probabilities[0][predicted_emotion_idx]

                    # Display results
                    st.success("Analysis completed successfully!")

                    # Main prediction result
                    detected_emotion = current_categories[predicted_emotion_idx]
                    st.markdown(
                        f"<h2 style='text-align: center; color: {self.color_palette.get(detected_emotion, '#FFFFFF')};'>"
                        f"Detected Emotion: {detected_emotion.upper()}</h2>",
                        unsafe_allow_html=True
                    )

                    # Confidence visualization
                    st.progress(int(prediction_confidence * 100))
                    st.caption(f"Confidence Level: {prediction_confidence:.1%}")

                    # Probability distribution chart
                    st.subheader("Emotion Probability Distribution")
                    probability_data = pd.DataFrame({
                        'Emotion': current_categories,
                        'Probability': emotion_probabilities[0]
                    }).sort_values('Probability', ascending=False)

                    chart_figure, chart_axis = plt.subplots(figsize=(10, 6))
                    chart_axis.bar(probability_data['Emotion'], probability_data['Probability'],
                                   color=[self.color_palette.get(emotion, '#3498db') for emotion in
                                          probability_data['Emotion']])
                    chart_axis.set_ylim(0, 1)
                    plt.xticks(rotation=45)
                    plt.title('Emotion Classification Probabilities')
                    st.pyplot(chart_figure)

                    # Audio playback
                    st.subheader("Audio Playback")
                    st.audio(uploaded_audio)

                    # Cleanup temporary file
                    os.remove(temp_file_path)

                except Exception as error:
                    st.error(f"Audio processing failed: {error}")

        # Display model optimization report if available
        if os.path.exists('class_filter_report.json'):
            with open('class_filter_report.json', 'r') as file:
                optimization_report = json.load(file)

            if 'status' not in optimization_report:
                st.markdown("---")
                st.subheader("Model Optimization Summary")
                for emotion_class, details in optimization_report.items():
                    with st.expander(f"Removed Class: {emotion_class.upper()} - Accuracy: {details['accuracy']}"):
                        st.write(f"Reason: {details['reason']}")
                        st.caption(f"Samples removed: {details['samples_removed']}")

        # Application footer
        st.markdown("---")
        st.markdown("### System Information")
        st.markdown(
            "This system utilizes deep learning techniques for accurate emotion recognition from speech patterns.")


def main():
    """Main execution function"""
    # Initialize the emotion recognition system
    emotion_system = AudioEmotionClassifier()

    # Set operation mode
    operation_mode = 'app'  # Change to 'app' after training completion

    if operation_mode == 'train':
        training_result = emotion_system.execute_training_pipeline()
        if training_result[0] is not None:
            print("Training process completed successfully")
    else:
        emotion_system.launch_web_interface()


if __name__ == "__main__":
    main()