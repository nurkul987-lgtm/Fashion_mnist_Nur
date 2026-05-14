import streamlit as st
import requests
from PIL import Image

st.title('Fashion MNIST Predictor')

uploaded_file = st.file_uploader(
    'загрузи фото',
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image,  width=250)

    if st.button('Predict'):

        files = {
            'file': uploaded_file.getvalue()
        }

        response = requests.post(
            'http://127.0.0.1:8001/predict/',
            files=files
        )

        if response.status_code == 200:

            result = response.json()

            st.success(f"Prediction: {result['Answer']}")

        else:
            st.error('API error')