# Real-Time Emotion Recognition using CNN & OpenCV

A real-time facial emotion recognition system built using **Convolutional Neural Networks (CNN)** and **OpenCV**. The system detects human faces from a webcam feed and predicts the corresponding facial emotion **in real time**.

This project demonstrates an **end-to-end machine learning workflow**: data preprocessing, model training, evaluation, and live deployment using a webcam.

---

## Features

- Real-time face detection using Haar Cascade Classifier  
- Emotion classification using a CNN trained on grayscale facial images  
- Live webcam emotion prediction  
- Lightweight and fast inference  
- Fully runnable on a local machine (CPU-based)  

---

## Emotions Detected

The model predicts **7 facial emotions**:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Jupyter Notebook  

---

## Model Details

- **Input shape:** 48 × 48 grayscale images  
- **Architecture:** Convolutional Neural Network (CNN)  
- **Loss function:** Categorical Crossentropy  
- **Optimizer:** RMSprop  
- **Output:** Softmax (7 emotion classes)  

The model was trained on a facial emotion dataset with proper normalization and reshaping for CNN compatibility.

---

## Real-Time Emotion Detection

The webcam application:

1. Captures live video using OpenCV  
2. Detects faces using Haar Cascade  
3. Preprocesses detected faces  
4. Feeds them into the trained CNN model  
5. Displays predicted emotion on the screen  

---

## How to Run the Project

## 1️⃣ **Clone the Repository**  

git clone https://github.com/PrasadWijesinghe/Real-Time-Emotion-Recognition.git
cd Real-Time-Emotion-Recognition

## 2️⃣ Create & Activate Virtual Environment

python -m venv emotion_env
.\emotion_env\Scripts\Activate.ps1   # Windows PowerShell

## 3️⃣ Install Dependencies

pip install -r requirements.txt

## 4️⃣ Run Webcam Emotion Detection

python .\webcam_emotion.py




##⚠️ Limitations

Sensitive to lighting conditions

Performance may vary with different face angles

Haar cascade face detection is less accurate compared to modern detectors

Emotion prediction depends on dataset quality and diversity

