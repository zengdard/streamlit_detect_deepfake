import os
import requests
import keras
from keras_nlp.tokenizers import UnicodeCodepointTokenizer
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np



os.environ["HUGGINGFACE_TOKEN"] = "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"


def download_file(url, local_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def load_keras_model_from_hub(model_id):
    model_url = f"https://huggingface.co/{model_id}/resolve/main/TALEYRAND.h5"
    local_path = "model_casia_run1.h5"
    download_file(model_url, local_path)
  

st.set_page_config(page_title="Fake Domain Detector", layout="wide")

st.title("Fake Domain Detector (Beta) - StendhalGPT Security")
# Préparer l'image
def prepare_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 128, 128, 3)
    return image

# Fonction pour hachurer l'image
def apply_hatching(image, percentage):
    hatched_image = np.array(image)
    h, w, _ = hatched_image.shape
    mask = np.random.choice([0, 1], size=(h, w), p=[1 - percentage, percentage])
    hatched_image[mask == 1] = [255, 0, 0]  # Couleur rouge pour le hachurage
    return hatched_image

# Interface Streamlit
st.title("Détection de fausses images")

# Charger l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try: 
        model = load_model('model_casia_run1.h5')
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    except:
        load_keras_model_from_hub('nielzac/private_fake')

    model = load_model('TALEYRAND.h5',compile=False)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_column_width=True)

    # Prétraitement de l'image
    preprocessed_image = prepare_image(image)

    # Prédiction
    prediction = model.predict(preprocessed_image)
    fake_percentage = prediction[0][1] * 100
    st.write(f"Probabilité d'être fausse : {fake_percentage:.2f}%")

    # Appliquer le hachurage
    hatched_image = apply_hatching(image, fake_percentage / 100)
    st.image(hatched_image, caption="Image avec hachurage", use_column_width=True)
