import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

st.title('Fashion MNIST Predictor')

model = torch.load('model_fashion_nur.pth', map_location='cpu')
model.eval()

classes = [
    'T-shirt/top','Trouser','Pullover','Dress','Coat',
    'Sandal','Shirt','Sneaker','Bag','Ankle boot'
]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader('загрузи фото', type=['png','jpg','jpeg'])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, width=250)

    if st.button('Predict'):

        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()

        st.success(f'Prediction: {classes[pred]}')