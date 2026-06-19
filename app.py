import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from fpdf import FPDF
import cv2
import matplotlib.cm as cm

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="DR Detection System",
    page_icon="👁️",
    layout="wide"
)

# ==================================
# DARK MODE
# ==================================

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:

    bg_color = "#0e1117"
    text_color = "white"
    card_color = "#262730"

else:

    bg_color = "#f4f8fb"
    text_color = "black"
    card_color = "white"

# ==================================
# CUSTOM CSS
# ==================================

st.markdown(f"""
<style>

.main {{
    background-color:{bg_color};
    color:{text_color};
}}

.title {{
    font-size:40px;
    font-weight:bold;
    color:#00b4d8;
    text-align:center;
}}

.card {{
    background-color:{card_color};
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.1);
    margin-top:20px;
}}

.stButton>button {{
    background-color:#0077b6;
    color:white;
    border-radius:10px;
    height:50px;
    width:220px;
    font-size:18px;
}}

.footer {{
    text-align:center;
    margin-top:50px;
    font-size:18px;
    color:#00b4d8;
    font-weight:bold;
}}

</style>
""", unsafe_allow_html=True)

# ==================================
# GRAD-CAM FUNCTION
# ==================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(
                last_conv_layer_name
            ).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    ) / tf.math.reduce_max(
        heatmap
    )

    return heatmap.numpy()

# ==================================
# LOGIN SYSTEM
# ==================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Doctor Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "doctor" and password == "1234":

            st.session_state.logged_in = True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error(
                "Wrong Username or Password"
            )

else:

    # ==================================
    # TITLE
    # ==================================

    st.markdown(
        '<p class="title">👁️ DR Detection Dashboard</p>',
        unsafe_allow_html=True
    )

    st.write(
        "AI Medical Diagnosis Platform"
    )

    st.divider()

    # ==================================
    # ABOUT DISEASE
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🩺 About Disease")

    st.write("""
Diabetic Retinopathy is a diabetes complication 
that affects the eyes. It occurs due to damage 
to the blood vessels in the retina.

Early detection can help prevent vision loss 
and blindness. AI systems can assist doctors 
by analyzing retina images quickly and accurately.
""")

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ==================================
    # LOAD MODEL
    # ==================================

    model = tf.keras.models.load_model(
        "final_dr_detection_efficientnetb3.keras"
    )

    # ==================================
    # SIDEBAR DASHBOARD
    # ==================================

    st.sidebar.title(
        "🩺 Doctor Dashboard"
    )

    st.sidebar.success(
        "System Online"
    )

    st.sidebar.info(
        "Model: Retina AI"
    )

    # ==================================
    # PATIENT INFORMATION
    # ==================================

    patient_name = st.sidebar.text_input(
        "Patient Name"
    )

    patient_age = st.sidebar.number_input(
        "Patient Age",
        1,
        100
    )

    patient_weight = st.sidebar.number_input(
        "Patient Weight (kg)",
        1,
        300
    )

    blood_pressure = st.sidebar.text_input(
        "Blood Pressure"
    )

    # ==================================
    # FILE UPLOAD
    # ==================================

    uploaded_file = st.file_uploader(
        "📤 Upload Retina Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        # ==================================
        # SHOW IMAGE
        # ==================================

        with col1:

            img = Image.open(uploaded_file)

            st.image(
                img,
                caption="Uploaded Retina Image",
                use_container_width=True
            )

        # ==================================
        # PREDICTION SECTION
        # ==================================

        with col2:

            if st.button(
                "🔍 Predict Disease"
            ):

                # ==================================
                # IMAGE PREPROCESSING
                # ==================================

                img_resized = img.resize(
                    (300, 300)
                )

                img_array = image.img_to_array(
                    img_resized
                )

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                img_array = img_array / 255.0

                # ==================================
                # PREDICTION
                # ==================================

                prediction = model.predict(
                    img_array
                )

                confidence = float(
                    prediction[0][0]
                ) * 100

                # ==================================
                # RESULT LOGIC
                # ==================================

                if confidence >= 78:

                    predicted_class = (
                        "Healthy Retina"
                    )

                else:

                    predicted_class = (
                        "Disease Detected"
                    )

                # ==================================
                # RESULT CARD
                # ==================================

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.success(
                    f"Prediction: {predicted_class}"
                )

                st.warning(
                    f"Confidence: {confidence:.2f}%"
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                # ==================================
                # REAL AI HEATMAP
                # ==================================

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "🔥 AI Heatmap"
                )

                st.write("""
The heatmap highlights the regions 
where the AI model focused while 
analyzing the retina image.
""")

                # Generate Heatmap

                heatmap = make_gradcam_heatmap(
                    img_array,
                    model,
                    "top_conv"
                )

                # Convert Original Image

                img_np = np.array(img_resized)

                # Resize Heatmap

                heatmap = cv2.resize(
                    heatmap,
                    (
                        img_np.shape[1],
                        img_np.shape[0]
                    )
                )

                # Convert Heatmap to RGB

                heatmap = np.uint8(
                    255 * heatmap
                )

                jet = cm.get_cmap("jet")

                jet_colors = jet(
                    np.arange(256)
                )[:, :3]

                jet_heatmap = jet_colors[
                    heatmap
                ]

                jet_heatmap = image.array_to_img(
                    jet_heatmap
                )

                jet_heatmap = jet_heatmap.resize(
                    (
                        img_np.shape[1],
                        img_np.shape[0]
                    )
                )

                jet_heatmap = image.img_to_array(
                    jet_heatmap
                )

                # Superimpose Heatmap

                superimposed_img = (
                    jet_heatmap * 0.4
                ) + img_np

                superimposed_img = image.array_to_img(
                    superimposed_img
                )

                # Display Heatmap

                st.image(
                    superimposed_img,
                    caption="AI Heatmap Visualization",
                    use_container_width=True
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

                # ==================================
                # RESULT TEXT
                # ==================================

                result_text = f"""
Patient Name: {patient_name}

Patient Age: {patient_age}

Patient Weight: {patient_weight} kg

Blood Pressure: {blood_pressure}

Prediction: {predicted_class}

Confidence: {confidence:.2f}%
"""

                # ==================================
                # DOWNLOAD TXT RESULT
                # ==================================

                st.download_button(

                    label="📥 Download Result",

                    data=result_text,

                    file_name="medical_result.txt",

                    mime="text/plain"
                )

                # ==================================
                # PDF REPORT
                # ==================================

                pdf = FPDF()

                pdf.add_page()

                pdf.set_font(
                    "Arial",
                    size=16
                )

                pdf.cell(
                    200,
                    10,
                    txt="Medical Report",
                    ln=True
                )

                pdf.ln(10)

                pdf.set_font(
                    "Arial",
                    size=12
                )

                pdf.multi_cell(
                    0,
                    10,
                    result_text
                )

                pdf.output(
                    "report.pdf"
                )

                with open(
                    "report.pdf",
                    "rb"
                ) as file:

                    st.download_button(

                        label="📄 Download PDF Report",

                        data=file,

                        file_name="report.pdf",

                        mime="application/pdf"
                    )

    st.divider()

    # ==================================
    # FOOTER
    # ==================================

    st.markdown(
        """
<div class="footer">
✨ Created By Shrouk Nasr Mahmoud ✨
</div>
""",
        unsafe_allow_html=True
    )

    # ==================================
    # LOGOUT
    # ==================================

    if st.button("Logout"):

        st.session_state.logged_in = False

        st.rerun()