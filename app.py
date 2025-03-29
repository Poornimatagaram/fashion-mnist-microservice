import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import joblib

# Load the model
model = joblib.load("fashion_mnist_model.joblib")
model.eval()

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to predict the class of the image
def predict_image(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit UI
st.title("Fashion-MNIST Image Classifier")

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    # Open and display the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict the class
    prediction = predict_image(img)
    
    # Display the prediction
    st.write(f"Predicted Class: {prediction}")
    st.write(f"Class Label: {['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'][prediction]}")
