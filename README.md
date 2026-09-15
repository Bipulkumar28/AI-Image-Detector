# 🔍 AI-Generated vs. Real Image Discriminator

A deep learning system built using **EfficientNetB0 (Transfer Learning)** to classify images as either **Real Photographs** or **AI-Generated Imagery**, featuring **Grad-CAM visual explainability** and an interactive **Streamlit web application**.

---

## 📌 Project Overview
* **Model Backbone:** EfficientNetB0 (Pretrained on ImageNet)
* **Classifier Head:** GlobalAveragePooling2D → BatchNormalization → Dropout (0.4) → Sigmoid Dense Unit
* **Performance:** 95.36% Test Accuracy | 0.9936 ROC-AUC
* **Explainability:** Grad-CAM Heatmaps (targeting `top_activation` layer)
* **Frontend:** Interactive Streamlit Dashboard


## ⚙️ Installation & Setup

### 1. Clone or Open Project Directory
Open your terminal or PowerShell and navigate to your project root folder:

cd path/to/AI_Image_Detector

2. Create and Activate a Virtual Environment
On Windows (PowerShell):

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
(If execution is blocked by policy, run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process first)

On macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
3. Install Required Packages
Run the following command to install all necessary dependencies:

Bash
python -m pip install --upgrade pip
python -m pip install tensorflow streamlit opencv-python pillow numpy matplotlib seab

How to Run the Application
Launching the Streamlit UI
Run the web application directly through Python:

Bash
python -m streamlit run app.py
