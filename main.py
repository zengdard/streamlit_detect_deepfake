import os
import requests
import keras
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

from PIL import Image, ImageChops, ImageEnhance
class_names = ['fake', 'real']
os.environ["HUGGINGFACE_TOKEN"] = "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"

image_size = (128, 128)

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

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

st.set_page_config(page_title="Fake Domain Detector", layout="wide")

st.title("Fake Domain Detector (Beta) - StendhalGPT Security")

# Charger l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        model = load_model('model_casia_run1.h5')
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    except:
        load_keras_model_from_hub('Nielzac/Altered_Picture_Model')

    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_column_width=True)
    
    image.save("chemin_de_sauvegarde.jpg")
    model = load_model('model_casia_run1.h5', compile=False)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
   
    image = prepare_image("chemin_de_sauvegarde.jpg")
    #image = image.reshape(-1, 128, 128, 3)
    image2 = np.reshape(image, (-1, 128, 128, 3))
    print(image2.shape)
    y_pred = model.predict(image2)
    # Prédiction
    
    fake_percentage = prediction[0][1] * 100
    st.write(f"Probabilité d'être fausse : {fake_percentage:.2f}%")

    # Appliquer le hachurage
    hatched_image = apply_hatching(prepared_image, fake_percentage / 100)
    st.image(hatched_image, caption="Image avec hachurage", use_column_width=True)
