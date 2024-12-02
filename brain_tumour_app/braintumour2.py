"""
Brain Tumour Classification
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# ---------------------------------------------------------------------------------------------------------------#
#                                                   PAGE SETTINGS                                               #
# ---------------------------------------------------------------------------------------------------------------#

# Setting page configuration
st.set_page_config(
    page_title="Brain Tumour Detection and Classification",
    page_icon=":hospital:",
    layout="wide",
)
st.markdown("<h1 style='text-align: center; color: black;'>Brain Tumour Detection and Classification</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------#
#                                                 FUNCTIONS & LOGIC                                              #
# ---------------------------------------------------------------------------------------------------------------#

def load_model():
    """
    Load the TensorFlow Lite model and initialize the interpreter.
    """
    try:
        model_path = r'D:\TY\Sem5\AI\BRAIN-TUMOR-DETECTION-USING-NEURAL-NETWORK\brain_tumour_app\model\model.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None


def predict(image, interpreter):
    """
    Predict the type of brain tumour based on the input image.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess image
    image = np.array(image.resize((150, 150)), dtype=np.float32)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = np.array(output_data[0])
    result = probabilities.argmax()

    labels = {0: 'Glioma Tumour', 1: 'Meningioma Tumour', 2: 'No Tumour', 3: 'Pituitary Tumour'}
    pred = labels[result]

    # Generate message and recommendations
    recommendation = "Proceed to the neurosurgery department for further consultation."
    if result == 0:
        message = f"Abnormal cell growth in the glial cells. {recommendation}"
    elif result == 1:
        message = f"Abnormal cell growth in the meninges. {recommendation}"
    elif result == 2:
        message = "No tumour detected."
    elif result == 3:
        message = f"Abnormal cell growth in the pituitary gland. {recommendation}"
    else:
        message = "Unknown classification."

    return pred, message


def main():
    """
    Main function for the Streamlit application.
    """
    interpreter = load_model()
    if interpreter is None:
        return

    file_uploaded = st.file_uploader("Upload a brain MRI image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if file_uploaded is not None:
        try:
            # Process the uploaded image
            file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            image = Image.open(file_uploaded)

            col1, col2 = st.columns([1, 1])

            with st.spinner("Processing..."):
                with col1:
                    st.image(opencv_image, channels="RGB", width=300, caption="Uploaded Image")
                with col2:
                    predictions, message = predict(image, interpreter)
                    st.success("Classification Complete!")
                    st.markdown(f"<h4 style='color: black;'>{predictions}</h4>", unsafe_allow_html=True)
                    st.write(message)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# ---------------------------------------------------------------------------------------------------------------#
#                                                   FOOTER                                                      #
# ---------------------------------------------------------------------------------------------------------------#

# st.warning(
#     """
#     Disclaimer: This application is intended for educational purposes only and should not be used for self-diagnosis.
#     Consult a medical professional for real-life cases.
#     """
# )

if __name__ == "__main__":
    main()
