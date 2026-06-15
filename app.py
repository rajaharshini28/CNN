import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
model = load_model("mnist_cnn_model.h5")

st.title("✍️ Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = image.convert("L")

    image = image.resize((28,28))

    img_array = np.array(image)

    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    st.image(image,
             caption="Uploaded Image")

    st.success(
        f"Predicted Digit: {digit}"
    )