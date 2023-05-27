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


POLICE = 'TypoSlab Irregular Demo.otf'


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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def apply_gradient_and_text(image, percentage, fake_score):
    # Convert image to numpy array
    image_array = np.array(image)

    # Calculate the height of the part to filter
    height, width, _ = image_array.shape
    filter_height = int(height * fake_score)

    # Create a mask with the same width and height as the original image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Calculate the start and end values for the gradient
    start_value = int(255 * 0.5)  # 50% opacity
    end_value = 0  # 0% opacity

    # Calculate the values for each pixel in the mask
    for i in range(filter_height, height):
        mask[i] = np.interp(i, [filter_height, height], [start_value, end_value])

    # Create a colored image (blue) with the same size as the original image
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    color_image[:,:] = (255, 0, 0)  # Blue color

    # Blend the original image with the color image using the mask
    blended_image = cv2.bitwise_and(color_image, color_image, mask=mask)
    blended_image = cv2.addWeighted(image_array, 1, blended_image, 1, 0)

    # Convert blended image to a Pillow image
    blended_image_pillow = Image.fromarray(blended_image)

    # Specify the font and size
    font = ImageFont.truetype("arial", size=75)

    # Create an ImageDraw object
    draw = ImageDraw.Draw(blended_image_pillow)

    # Convert fake score to text and calculate the text's coordinates
    text = "FAKE {:.1f}%".format(fake_score * 100)
    text_width, text_height = draw.textsize(text, font)
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 4

    # Draw the text on the image in white
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    return blended_image_pillow


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

st.set_page_config(page_title="Fake Photo Identifier", layout="wide")

st.title("Fake Photo Identifier (Beta) - StendhalGPT Gogh")

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
    #st.write(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

    # Appliquer le hachurage
    hatched_image = apply_fake_filter(image3,np.amax(y_pred), class_names[y_pred_class])

    # Display the image
    st.image(hatched_image, use_column_width=False)
    
    st.markdown("""
    ## Source des données
    Les données utilisées pour l'entraînement de ce modèle proviennent de l'ensemble de données CASIA disponible sur Kaggle, 
    qui a été partagé par Sophatvathana. L'ensemble de données peut être consulté à l'adresse suivante : 
    [CASIA Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)
    """)
