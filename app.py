import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import matplotlib.cm as cm

# Page configuration
st.set_page_config(
    page_title="AI vs Real Image Discriminator",
    page_icon="",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "best_discriminator_model.keras")

@st.cache_resource
def load_discriminator():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure training is complete.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_discriminator()

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_activation"):
    # Retrieve the inner EfficientNetB0 backbone and classification head layers
    try:
        base_model = model.get_layer("efficientnetb0")
    except KeyError:
        base_model = model

    # Create a sub-model that maps raw images to the target conv layer output
    conv_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer(last_conv_layer_name).output
    )

    # Extract top head layers from the main model
    gap_layer = model.get_layer("global_average_pooling2d")
    bn_layer = model.get_layer("batch_normalization")
    dropout_layer = model.get_layer("dropout")
    classifier_layer = model.get_layer("classifier")

    # Record operations under GradientTape
    with tf.GradientTape() as tape:
        # 1. Forward pass through base conv model
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        
        # 2. Forward pass through remaining top head layers
        x = gap_layer(conv_outputs)
        x = bn_layer(x, training=False)
        x = dropout_layer(x, training=False)
        predictions = classifier_layer(x)
        
        class_channel = predictions[0]

    # Compute gradients of class score with respect to feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map channels by computed gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep positive contributions and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    # Rescale heatmap to 0-255
    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize using Jet colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = (jet_heatmap * 255).astype(np.uint8)

    # Superimpose heatmap on original image
    original_np = np.array(original_img)
    superimposed = cv2.addWeighted(original_np, 1 - alpha, jet_heatmap, alpha, 0)
    return superimposed

# --- UI HEADER ---
st.title("🔍 AI-Generated vs Real Image Discriminator")
st.markdown("A Transfer Learning Convolutional Neural Network (EfficientNetB0) for detecting synthetic AI imagery with Grad-CAM explainability.")

st.sidebar.header("Configuration & Info")
st.sidebar.markdown("""
**Model Details:**
- **Backbone:** EfficientNetB0
- **Input Size:** 224 × 224
- **Test Accuracy:** 95.36%
- **ROC-AUC:** 0.9936

**Label Conventions:**
- **Prob < 0.50:** AI-Generated Image
- **Prob ≥ 0.50:** Real Photograph
""")

# --- MAIN WORKFLOW ---
uploaded_file = st.file_uploader("Upload an image to inspect...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    col1, col2 = st.columns(2)

    # Display raw image
    raw_image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Original Image")
        st.image(raw_image, use_container_width=True)

    # Preprocess image
    img_resized = raw_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array_batch = np.expand_dims(img_array, axis=0)

    # Inference
    with st.spinner("Analyzing high-frequency image features..."):
        prob_real = float(model.predict(img_array_batch, verbose=0)[0][0])
        prob_ai = 1.0 - prob_real

    # Determine class
    if prob_real >= 0.5:
        prediction_label = "REAL PHOTOGRAPH"
        confidence = prob_real * 100
        status_color = "success"
    else:
        prediction_label = "AI-GENERATED IMAGE"
        confidence = prob_ai * 100
        status_color = "error"

    # Display prediction results
    with col2:
        st.subheader("Prediction Results")
        if status_color == "success":
            st.success(f"**Classification:** {prediction_label}")
        else:
            st.error(f"**Classification:** {prediction_label}")

        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        st.write("---")
        st.write("**Probability Distribution:**")
        st.progress(prob_ai, text=f"AI-Generated Probability: {prob_ai * 100:.2f}%")
        st.progress(prob_real, text=f"Real Photograph Probability: {prob_real * 100:.2f}%")

    # Grad-CAM Section
    st.write("---")
    st.subheader("💡 Model Explainability (Grad-CAM)")
    st.markdown("The heatmap highlights spatial regions and visual features that most heavily influenced the neural network's final classification.")

    show_gradcam = st.checkbox("Generate Grad-CAM Heatmap", value=True)

    if show_gradcam:
        try:
            heatmap = make_gradcam_heatmap(img_array_batch, model, last_conv_layer_name="top_activation")
            superimposed_img = overlay_heatmap(heatmap, raw_image)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(superimposed_img, caption="Grad-CAM Activation Overlay", use_container_width=True)
            with g_col2:
                st.info("""
                **How to interpret this visualization:**
                - **Warm regions (Red/Yellow):** Indicate primary regions of interest (e.g., lighting anomalies, unnatural textures, spatial blur, structural mismatches).
                - **Cool regions (Blue/Purple):** Regions largely ignored by the classification head.
                """)
        except Exception as e:
            st.warning(f"Could not render Grad-CAM heatmap: {str(e)}")