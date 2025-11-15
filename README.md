# 😴 Drowsiness Detection Model  

### 🚗 Internship Project | Machine Learning | Computer Vision  

A machine learning–based system that detects **driver drowsiness** in real time from **images** or **videos**, capable of identifying multiple people, predicting their **awake/sleeping state**, and estimating their **ages**.  
It provides visual feedback using bounding boxes and pop-up alerts, helping improve road safety through early detection of fatigue.

---

## 🧩 Problem Statement

Driver drowsiness and fatigue are major causes of road accidents worldwide.  
Long hours of driving can lead to micro-sleeps or reduced alertness, which often go unnoticed until it’s too late.  

The objective of this project is to develop an **automated system** that:
- Detects whether a driver (or passenger) is **awake or asleep** from live images or video streams.
- Handles **multiple people** simultaneously.
- Highlights **sleeping persons in red**, **awake persons in green**.
- Optionally predicts **age** for each detected individual.
- Displays a **pop-up alert** summarizing the number of sleeping individuals and their ages.

---

## 🧠 Dataset Description

### 1️⃣ **MRL Eye Dataset**
- Source: [MRL Eye Dataset on Kaggle](https://www.kaggle.com/datasets/tom99763/mrl-eye-dataset)
- Total Images: ~85,000  
- Classes: **Open Eyes** and **Closed Eyes**
- Format: PNG images (grayscale, 24x24)
- Usage: To train the CNN model to classify eyes as open or closed.

### 2️⃣ **Custom Test Data**
- Additional driver and passenger images/videos were used for validation.
- These simulate real-world driving conditions (different lighting, head angles, etc.).

---

## ⚙️ Methodology

### Step 1: **Data Preprocessing**
- MRL dataset organized into:

```
data/
└── eyes/
├── train/
│ ├── open/
│ └── closed/
└── val/
├── open/
└── closed/
```

- All images resized to **48×48 pixels** and normalized (0–1 range).
- Dataset split into **80% training** and **20% validation**.

---

### Step 2: **Model Training (Eye State CNN)**

- Architecture:  
- **3 Convolutional layers** with ReLU activation  
- **MaxPooling** after each  
- **Dropout** (0.3) for regularization  
- **Dense layers** leading to a **softmax output (2 classes)**  
- Framework: **TensorFlow / Keras**
- Optimizer: **Adam (lr = 0.001)**
- Epochs: **12**
- Batch size: **64**
- Output:  
- Model → `models/eye_state_cnn.h5`  
- Labels → `models/eye_state_labels.json`  

Training command:
```bash
cd training
python train_eye_state.py
```

### Step 3: **Integration with MediaPipe FaceMesh**

Used MediaPipe Face Mesh to detect facial landmarks (468 points).

Extracted eye landmarks for each detected face.

Computed Eye Aspect Ratio (EAR) to measure openness of eyes.

Combined EAR thresholding and CNN predictions for improved accuracy.


### Step 4: **Age Prediction (Optional)**

Integrated DeepFace to estimate approximate age for each detected face.

Only runs if the “Enable Age Prediction” option is active in the GUI.


### Step 5: **User Interface (Streamlit GUI)**

Built using Streamlit for a clean, user-friendly experience.

Features:

Upload images or videos.

Adjustable EAR threshold.

Option to enable/disable age prediction.

Real-time preview with color-coded bounding boxes.

Pop-up message summarizing sleeping persons and ages.


# 🧮 Results and Analysis

## ✅ Model Performance

| Metric   | Training | Validation |
| -------- | -------- | ---------- |
| Accuracy | 96.1%    | 94.8%      |
| Loss     | 0.12     | 0.18       |


### 🧩 Project Structure

```
📦 drowsiness_detection
│
├── app.py                     # Streamlit GUI
├── utils/
│   ├── vision.py              # Frame processing logic
│
├── models/
│   ├── eye_state_cnn.h5       # Trained CNN model
│   ├── eye_state_labels.json  # Label info
│
├── training/
│   ├── prepare_mrl.py         # Dataset prep script
│   ├── train_eye_state.py     # Model training script
│
├── data/
│   └── eyes/                  # Train/Val data folders
│
├── requirements.txt
└── README.md
```

### 📈 Key Learning Outcomes

✅ Dataset preprocessing & augmentation
✅ CNN-based binary image classification
✅ Real-time face landmark detection
✅ Integration of DeepFace + MediaPipe
✅ Streamlit GUI for live visualization
✅ Full-stack ML project deployment
---

##  Data and Visual output Link

[Drive Link](https://drive.google.com/drive/folders/1m4kGJfVO1-9yO3rvpyFMH-wAeOMqsONW?usp=sharing)



