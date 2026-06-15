# CNN Handwritten Digit Recognition

## Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits.

## Dataset
MNIST Dataset

## Technologies
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow

## Architecture

Input Image (28x28)
↓
Conv2D(32)
↓
MaxPooling
↓
Conv2D(64)
↓
MaxPooling
↓
Flatten
↓
Dense(128)
↓
Output (0-9)

## Run

Train:

```bash
python train.py
```

Start App:

```bash
streamlit run app.py
```

## Accuracy

98%+