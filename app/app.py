import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import io
import soundfile as sf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd # Added this import

# --- Streamlit App Setup ---
st.set_page_config(page_title="Emotion Recognition from Speech", layout="centered")
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file to recognize the emotion expressed.")

# --- Global Variables (from previous Colab session) ---
# These are assumed to be initialized from the previous notebook cells.
# If running this standalone, you'd need to load/re-initialize them.
# For this demonstration within Colab, we rely on the kernel state.

global model, scaler, label_encoder, X_train_reshaped

# Load the trained CNN model
# The model was saved as 'best_cnn_model.keras' during callback training
# Assuming `model` variable in kernel refers to the trained CNN model
# And it has the best weights restored by EarlyStopping.
# If running standalone, you would load it like: model = tf.keras.models.load_model('best_cnn_model.keras')
# For this context, we'll assume the 'model' variable already holds the loaded model with best weights.
# Let's re-load it to be safe, especially if the kernel state changed or this is run after other modifications.
try:
    model = tf.keras.models.load_model('best_cnn_model.keras')
except Exception as e:
    st.error(f"Error loading the CNN model: {e}. Please ensure 'best_cnn_model.keras' exists and was saved correctly.")
    st.stop()

# Assuming `scaler` and `label_encoder` are available from previous executed cells
# If running standalone, you'd need to save/load these as well, e.g., using joblib or pickle.
if 'scaler' not in globals():
    st.error("StandardScaler object `scaler` not found. Please ensure feature normalization was run.")
    st.stop()
if 'label_encoder' not in globals():
    st.error("LabelEncoder object `label_encoder` not found. Please ensure label encoding was run.")
    st.stop()

# --- Feature Extraction Function (Adapted) ---
def extract_mfcc_features_for_streamlit(audio_data, sr=22050, n_mfcc=40):
    """
    Extracts MFCC features from raw audio data (numpy array).
    """
    # Ensure audio is float type
    audio_data = audio_data.astype(np.float32)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# --- Streamlit File Uploader ---
uploaded_file = st.file_uploader("Choose an audio file... (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Analyzing emotion..."):
        try:
            # Read the uploaded WAV file directly into librosa
            audio_bytes = uploaded_file.read()
            # Use soundfile to read the audio data and sample rate
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

            # Extract features
            features = extract_mfcc_features_for_streamlit(audio_data, sr=sample_rate)

            # Normalize features using the *fitted* scaler
            # The scaler expects a 2D array: (n_samples, n_features)
            features_normalized = scaler.transform(features.reshape(1, -1))

            # Reshape for CNN model: (n_samples, timesteps, features_per_timestep)
            # Here, 1 sample, 40 MFCCs as timesteps, 1 feature per timestep
            features_reshaped = features_normalized[..., np.newaxis]

            # Make prediction
            predictions = model.predict(features_reshaped)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            predicted_emotion = label_encoder.inverse_transform([predicted_class_idx])[0]
            confidence = np.max(predictions) * 100

            st.success(f"Predicted Emotion: **{predicted_emotion.upper()}** (Confidence: {confidence:.2f}%) 🚀")

            st.subheader("Prediction Probabilities")
            # Create a dictionary for probabilities
            prob_dict = {label_encoder.classes_[i]: predictions[0][i] for i in range(len(label_encoder.classes_))}
            # Sort for better visualization
            sorted_prob = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            
            prob_df = pd.DataFrame(sorted_prob, columns=['Emotion', 'Probability'])
            st.dataframe(prob_df, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please try uploading a different audio file or ensure it's a valid WAV format.")

st.markdown("""
---
Created as part of a machine learning project for speech emotion recognition.
""")

!streamlit run app.py & npx localtunnel --port 8501
