import os
import requests
import keras
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

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
    model_url = f"https://huggingface.co/Nielzac/Altered_Picture_Model/resolve/main/model_casia_run1.h5"
    local_path = "model_casia_run1.h5"
    download_file(model_url, local_path)
import numpy as np
from PIL import Image, ImageDraw
def apply_hatching(image, percentage, fake_score):
    # Convertir l'image en tableau NumPy
    image_array = np.array(image)

    # Déterminer les dimensions de l'image
    height, width, _ = image_array.shape

    # Calculer la hauteur de la partie à filtrer
    filter_height = int(height * percentage)

    # Convertir le score de fausseté en texte
    text = f"{fake_score}"

    # Spécifier la police et la taille
    font = ImageFont.truetype("TypoSlab Irregular shadowed_demo.otf", size=40)

    # Créer un objet ImageDraw
    draw = ImageDraw.Draw(image)

    # Spécifier les coordonnées du texte
    y = height/2
    x = width/2

    # Dessiner le texte sur l'image avec l'opacité réduite
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Calculer la taille du filtre rouge proportionnellement au pourcentage
    filter_height = int(height * percentage)
    red_filter = Image.new("RGBA", (width, filter_height), (255, 0, 0, 128))

    # Créer une image masque blanche pour le reste de l'image
    white_mask = Image.new("RGBA", (width, height - filter_height), (255, 255, 255, 255))

    # Fusionner le filtre rouge et le masque blanc
    filter_image = Image.alpha_composite(red_filter, white_mask)

    # Fusionner le filtre avec l'image d'origine
    image_with_filter = Image.alpha_composite(image.convert("RGBA"), filter_image)

    return image_with_filter


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
        load_keras_model_from_hub('Altered_Picture_Model')

    image3 = Image.open(uploaded_file)
    image = image3
    #st.image(image, caption="Image originale", use_column_width=True)
    
    image.save("chemin_de_sauvegarde.jpg")
    print('#################OK')
    model = load_model('model_casia_run1.h5', compile=False)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
   
    image = prepare_image("chemin_de_sauvegarde.jpg")
    print(image.shape)
    #image = image.reshape(-1, 128, 128, 3)
    image2 = np.reshape(image, (-1, 128, 128, 3))
    print(image2.shape)
    y_pred = model.predict(image2)
    # Prédiction
    
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    st.write(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

    # Appliquer le hachurage
    hatched_image = apply_hatching(image3, np.amax(y_pred),np.amax(y_pred))
    st.image(hatched_image, caption="Image avec hachurage", use_column_width=True)
