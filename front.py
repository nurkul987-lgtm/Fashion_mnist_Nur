import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

st.title('Fashion MNIST Predictor')

# ---- MODEL CLASS (бул маанилүү!) ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# ---- LOAD MODEL ----
model = Net()
model.load_state_dict(torch.load('model_fashion_nur.pth', map_location='cpu'))
model.eval()

# ---- CLASSES ----
classes = [
    'T-shirt/top','Trouser','Pullover','Dress','Coat',
    'Sandal','Shirt','Sneaker','Bag','Ankle boot'
]

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# ---- UI ----
uploaded_file = st.file_uploader('загрузи фото', type=['png','jpg','jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, width=250)

    if st.button('Predict'):

        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()

        st.success(f'Prediction: {classes[pred]}')