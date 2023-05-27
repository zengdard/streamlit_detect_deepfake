import os
import requests
import keras
from keras_nlp.tokenizers import UnicodeCodepointTokenizer
from keras.models import load_model
import streamlit as st

os.environ["HUGGINGFACE_TOKEN"] = "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"


def download_file(url, local_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def load_keras_model_from_hub(model_id):
    model_url = f"https://huggingface.co/{model_id}/resolve/main/TALEYRAND.h5"
    local_path = "TALEYRAND.h5"
    download_file(model_url, local_path)
   






st.set_page_config(page_title="Fake Domain Detector", layout="wide")

st.title("Fake Domain Detector (Beta) - StendhalGPT Security")
st.markdown(
    """
    This service helps detect potentially fake domain names. The model is trained on a dataset of 50,000 sites,
    which represents around 2% of the available data. It's suitable for businesses and also integrated soon into StendhalGPT+.\n
    Only avalaible for .fr websites. \nExample : 'colis-livraison.fr', 'cnil-info.fr', 'antai-gov.fr', 'amendes-paiement.fr', 'leclerc.fr', 'hachette.fr'.  
    """
)

# Input domain names
domain_names = st.text_input("Enter a domain name:")

if st.button("Predict"):
    try: 
        model = load_model('TALEYRAND.h5')
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    except:
        load_keras_model_from_hub('nielzac/private_fake')

    model = load_model('TALEYRAND.h5',compile=False)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    


    tokenizer = UnicodeCodepointTokenizer(input_encoding="ISO 8859-1", output_encoding="ISO 8859-1")
    tokenized_domains_2 = []
    for name in [domain_names]:
        tokens = tokenizer.tokenize(name)
        tokenized_domains_2.append(tokens)

    X_new = keras.utils.data_utils.pad_sequences (tokenized_domains_2, maxlen=26, padding='post')

    # Faire une prédiction sur les noms de domaines
    y_pred = model.predict(X_new)
    
    # Display prediction
    st.write("Prediction score: ", y_pred)

    threshold = 0.5
   
    if y_pred[0] < threshold:
        st.write(f"The domain is likely to be genuine. (Prediction score: {y_pred[0][0]})")
        background_color = "green"
    else:
        st.write(f"The domain is likely to be fake. (Prediction score: {y_pred[0][0]})")
        background_color = "red"

    confidence_interval = 0.1
    if abs(y_pred[0] - threshold) < confidence_interval:
        st.write("Note: The prediction score is close to the threshold. The result may not be as reliable.")
        

# Définir les informations sur l'origine des données
    data_source = "Afnic, https://dl.red.flag.domains/ and others"
    # Définir l'adresse de contact
    contact_address = "contact@stendhalgpt.fr"
    st.write(f"Data source: {data_source}")
    st.write(f"Contact: {contact_address}")
    st.markdown(f'<style>body {{ background-color: {background_color}; }}</style>', unsafe_allow_html=True)
