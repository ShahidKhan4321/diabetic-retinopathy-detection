# 🩸 Diabetic Retinopathy Detection

An AI-powered web application that helps in the **early detection and classification of Diabetic Retinopathy (DR)** using **fundus retina images**.  
The system predicts the DR stage from the uploaded image and provides an intuitive interface for clinicians, researchers, and students.

---

## 🌟 Features
- 🖥️ **User-friendly Web App** built with **Streamlit**
- 🔍 Automatic **image preprocessing** (resizing, normalization)
- 🤖 **Deep Learning–based classifier** trained on the **APTOS 2019 dataset**
- 📊 Predicts **5 DR stages**:
  - **0 – No DR**
  - **1 – Mild DR**
  - **2 – Moderate DR**
  - **3 – Severe DR**
  - **4 – Proliferative DR**
- 📈 Displays model performance metrics (accuracy, F1-score)
- ⚡ Fast and lightweight — runs locally or in the cloud
- 🌐 API support via **Flask backend** for integration with other apps

---

## 🏗️ Project Architecture
User → Web UI (Streamlit) → Image Processor → Trained Model → Prediction → Results
↑
└─> Flask API (optional for integration)

---

## 📂 Project Structure
diabetic-retinopathy-detection/
│
├── app_streamlit.py # Streamlit web interface
├── app.py # Flask API
├── train_model.py # Model training script
├── model.pth # Trained model weights (~43 MB)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── screenshots/ # UI and prediction screenshots
└── sample_images/ # Few test retina images

---

## 🚀 Installation & Setup (Local)

### 1. Clone Repository
```bash
git clone https://github.com/ShahidKhan4321/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

## 2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate    # macOS/Linux

## 3. Install Dependencies
pip install -r requirements.txt

## 4. Run Streamlit Web App
streamlit run app_streamlit.py

## 5. (Optional) Run Flask API
python app.py
Then open the Index.html,if you’re using the API.


🌐 [Live Demo:](https://diabetic-retinopathy-detection-cnn.streamlit.app/)
🚀 Live Web App on Streamlit Cloud


📊 Model Details
Dataset: APTOS 2019 Blindness Detection
Model: CNN / Transfer Learning (e.g., ResNet / EfficientNet)
Framework: PyTorch
Accuracy: ~82.67% on validation set 

📝 Future Enhancements

🔹 Deploy as a full-scale cloud-hosted API

🔹 Add Grad-CAM heatmaps to visualize affected retinal regions

🔹 Improve accuracy with larger datasets and advanced architectures

🔹 Add support for real-time camera input

🔹 Integrate patient record management

