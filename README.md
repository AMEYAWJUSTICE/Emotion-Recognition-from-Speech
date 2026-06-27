# Speech Emotion Recognition Project

This project demonstrates how to build and deploy a Speech Emotion Recognition (SER) system using deep learning, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The system processes audio files, extracts relevant features, and predicts the emotion expressed in the speech.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Feature Extraction](#feature-extraction)
4.  [Preprocessing](#preprocessing)
5.  [Model Training](#model-training)
    -   [CNN Architecture](#cnn-architecture)
    -   [LSTM Architecture](#lstm-architecture)
6.  [Model Evaluation](#model-evaluation)
7.  [Streamlit Web Application](#streamlit-web-application)
8.  [Exported Files](#exported-files)
9.  [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
10. [Requirements](#requirements)

## Introduction
Speech Emotion Recognition (SER) is a challenging task that involves identifying human emotions from speech. This project uses the RAVDESS dataset, extracts MFCC features, and trains deep learning models to classify emotions. A Streamlit web application is also provided for interactive demonstration.

## Dataset
The project utilizes the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. This dataset contains 7356 files (speech and song) from 24 professional actors, vocalizing two lexically-matched statements in a neutral North American accent. The emotions covered are: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

**Data Source**: The dataset was downloaded via KaggleHub and is located at `/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24`.

## Feature Extraction
Mel-Frequency Cepstral Coefficients (MFCCs) are used as features. MFCCs are widely used in speech recognition and are robust to variations in speech. For each audio file, 40 MFCCs are extracted, and their mean over time is taken to create a fixed-size feature vector.

## Preprocessing
1.  **Feature Normalization**: The extracted MFCC features (`X`) are standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1. The normalized features are stored in `X_normalized`.
2.  **Label Encoding**: The categorical emotion labels (`y`) are converted into numerical labels using `LabelEncoder`. The encoded labels are stored in `y_encoded`.
3.  **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets using `train_test_split`, with stratification to maintain class distribution.
4.  **t-SNE Visualization**: t-SNE (t-Distributed Stochastic Neighbor Embedding) is applied to the training features to reduce dimensionality to 2D for visualization, helping to understand the separability of different emotion clusters.

## Model Training
Two deep learning models were built and trained for emotion classification:

### CNN Architecture
A 1D Convolutional Neural Network (CNN) is used to capture local patterns in the MFCC features. The architecture consists of:
-   Two `Conv1D` layers with `MaxPooling1D` for feature learning.
-   `Dropout` layers for regularization.
-   A `Flatten` layer to convert 2D feature maps to 1D.
-   `Dense` layers for classification, with a `softmax` output layer.

The input shape for the CNN is `(40, 1)`, treating each of the 40 MFCCs as a timestep with 1 feature.

### LSTM Architecture
A Long Short-Term Memory (LSTM) network is employed to capture sequential dependencies in the MFCC features. The architecture includes:
-   Two `LSTM` layers, with the first returning sequences.
-   `Dropout` layers for regularization.
-   `Dense` layers for classification, with a `softmax` output layer.

The input shape for the LSTM is also `(40, 1)`, similar to the CNN.

Both models are compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss.

### Training with Callbacks
The CNN model was further trained with `EarlyStopping` (patience=10 on `val_loss`) and `ModelCheckpoint` (`best_cnn_model.keras` saving the best model based on `val_accuracy`). This ensures optimal model selection and prevents overfitting.

## Model Evaluation
After training, both models are evaluated on the test set using:
-   **Test Loss and Accuracy**: Overall performance metrics.
-   **Confusion Matrix (for CNN)**: Visual representation of correct vs. incorrect classifications per class.
-   **Classification Report (for CNN and LSTM)**: Detailed metrics including precision, recall, and F1-score for each emotion class.

## Streamlit Web Application
A Streamlit application (`app.py`) has been created to provide an interactive interface for emotion recognition from uploaded audio files. Users can upload a `.wav` file, and the app will predict the emotion and display prediction probabilities.

## Exported Files
The following files have been exported for reusability and deployment:
-   `best_cnn_model.keras`: The trained CNN model with the best weights (saved during callback training).
-   `scaler.joblib`: The fitted `StandardScaler` object, essential for normalizing new audio features.
-   `label_encoder.joblib`: The fitted `LabelEncoder` object, needed to convert numerical predictions back to emotion labels.
-   `X_normalized.npy`: Normalized MFCC features of the entire dataset.
-   `y_encoded.npy`: Encoded labels of the entire dataset.
-   `requirements.txt`: A list of all Python packages and their versions used in this environment, crucial for reproducing the environment.
-   `app.py`: The Streamlit application code.

## How to Run the Streamlit App
1.  **Ensure all dependencies are installed** (refer to `requirements.txt`).
2.  **Make sure `app.py`, `best_cnn_model.keras`, `scaler.joblib`, and `label_encoder.joblib` are in the same directory.**
3.  **Execute the following command in your terminal or a Colab code cell:**
    ```bash
    streamlit run app.py & npx localtunnel --port 8501
    ```
    This command starts the Streamlit app and uses `localtunnel` to create a publicly accessible URL. Open this URL in your web browser to interact with the Speech Emotion Recognition app.

## Requirements
The `requirements.txt` file contains a detailed list of all necessary Python libraries. You can install them using pip:
```bash
pip install -r requirements.txt
