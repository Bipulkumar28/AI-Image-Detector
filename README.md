# 🔍 AI-Generated vs. Real Image Discriminator

A deep learning system using **EfficientNetB0 Transfer Learning** to classify images as **Real** or **AI-Generated**, with **Grad-CAM explainability** and an interactive **Streamlit web application**.

## 📌 Features

* **Model:** EfficientNetB0 (ImageNet pretrained)
* **Classifier:** GlobalAveragePooling → BatchNormalization → Dropout (0.4) → Sigmoid
* **Test Accuracy:** 95.36%
* **ROC-AUC:** 0.9936
* **Explainability:** Grad-CAM (`top_activation` layer)
* **Frontend:** Streamlit

## ⚙️ Installation

### 1. Create Virtual Environment

**Windows:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow streamlit opencv-python pillow numpy matplotlib seaborn
```

## 🚀 Run the Application

From the project directory:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 🧠 How It Works

**Upload Image → EfficientNetB0 → Prediction → Confidence Score → Grad-CAM Heatmap**

The model predicts whether an image is **Real** or **AI-Generated** and Grad-CAM highlights the regions that influenced the prediction.
