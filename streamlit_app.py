import streamlit as st
from googletrans import Translator
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re
import urllib.request
import os

# Configuración de la página con fondo animado
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

# Aplicar CSS personalizado para el fondo y diseño
st.markdown("""
<style>
    /* Fondo animado */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://64.media.tumblr.com/817c19affd93dc7dc145364acbb10331/8e4bb3b18c84e15f-60/s1280x1920/5cdcb9e6cb7edc05ab6994b12132f590033e7c0b.gifv");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Contenedor principal de Streamlit - transparente */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Rectángulo negro para el contenido */
    .content-box {
        background-color: rgba(0, 0, 0, 0.85);
        border-radius: 15px;
        padding: 10rem;
        margin: 0 auto;
        max-width: 1200px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Texto blanco para todo dentro del content-box */
    .content-box h1, 
    .content-box h2, 
    .content-box h3, 
    .content-box h4, 
    .content-box h5, 
    .content-box h6,
    .content-box p,
    .content-box label,
    .content-box div {
        color: white !important;
    }

    /* Personalización de componentes dentro del content-box */
    .content-box .stTextInput input, 
    .content-box .stTextArea textarea {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .content-box .stSelectbox select {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .content-box .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }

    /* Contenedor de resultados con texto negro */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-weight: bold;
        color: #000000 !important;
    }
    
    .result-box h3,
    .result-box p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Crear carpeta 'model' si no existe
if not os.path.exists('model'):
    os.makedirs('model')

# URLs de los modelos
MODEL_URLS = {
    "Inicial": {
        "model": "https://github.com/jgekko/sentio-app/raw/main/model/modelSENTIO.h5",
        "tokenizer": "https://github.com/jgekko/sentio-app/raw/main/model/tokenizer.pkl"
    },
    "Normal": {
        "model": "https://github.com/jgekko/sentio-app/raw/main/model/modelSENTIOfinal.keras",
        "tokenizer": "https://github.com/jgekko/sentio-app/raw/main/model/tokenizerSENTIOfinal.pkl"
    }
}

@st.cache_resource
def load_resources(model_type="Inicial"):
    # Descargar archivos si no existen
    model_filename = "modelSENTIO.h5" if model_type == "Inicial" else "modelSENTIOfinal.keras"
    tokenizer_filename = "tokenizer.pkl" if model_type == "Inicial" else "tokenizerSENTIOfinal.pkl"
    
    model_path = f'model/{model_filename}'
    tokenizer_path = f'model/{tokenizer_filename}'
    
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URLS[model_type]["model"], model_path)
    if not os.path.exists(tokenizer_path):
        urllib.request.urlretrieve(MODEL_URLS[model_type]["tokenizer"], tokenizer_path)
    
    # Cargar modelo según extensión
    if model_filename.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    else:  # .keras
        model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

# --- Interfaz de Usuario ---
st.markdown('<div class="content-box">', unsafe_allow_html=True)

# Selector de modelo
model_type = st.selectbox("Selecciona el modelo:", ["SENTIO v0", "SENTIO v1"])

# Cargar recursos según modelo seleccionado
try:
    model, tokenizer = load_resources(model_type)
except Exception as e:
    st.error(f"Error cargando recursos: {str(e)}")
    st.stop()

# Resto del código de funciones (igual que antes)
def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    return cleaned_texts

def preprocess_text(texts, tokenizer, max_len=50):
    text_seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(text_seq, maxlen=max_len, padding="post")

def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Error en traducción: {e}")
        return text

def predict_sentiment(text):
    try:
        text = [text]
        text = clean_text(text)
        text_padded = preprocess_text(text, tokenizer)
        
        y_prob = model.predict(text_padded, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        classes = ['Negative', 'Positive', 'Neutral']
        pred_class = classes[y_pred[0]]
        pred_prob = float(y_prob[0][y_pred[0]])
        
        emoji_map = {
            'Negative': '😠 Negativo',
            'Positive': '😊 Positivo', 
            'Neutral': '😐 Neutral'
        }
        
        return emoji_map[pred_class], pred_prob
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None, None

# Interfaz principal
st.title("🔍 Sentio - Análisis de Sentimientos")
    
st.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en español/inglés.
2. Selecciona el idioma del texto.
3. Haz clic en "Analizar Sentimiento".
""")
    
col1, col2 = st.columns([1, 1])
    
with col1:
    st.header("📝 Ingresa tu texto")
    language = st.selectbox("Idioma:", ["Español", "English"])
    user_input = st.text_area("Escribe aquí:", max_chars=50, height=100)
    analyze_btn = st.button("Analizar Sentimiento", type="primary")
    
with col2:
    st.header("📊 Resultado")
        
    if analyze_btn:
        if not user_input:
            st.warning("⚠️ Por favor ingresa texto")
        else:
            with st.spinner("Analizando..."):
                input_text = translate_to_english(user_input) if language == "Español" else user_input
                sentiment, confidence = predict_sentiment(input_text)
                
                if sentiment and confidence:
                    confidence_pct = round(confidence * 100, 2)
                        
                    if "Positivo" in sentiment:
                        bg_color = "#339b2f"
                    elif "Negativo" in sentiment:
                        bg_color = "#9b2f2f"
                    else:
                        bg_color = "#9b912f"
                            
                    st.markdown(f"""
                    <div class="result-box" style="background: {bg_color}">
                        <h3>Predicción (Modelo {model_type}):</h3>
                        <p style='font-size: 24px;'>{sentiment}</p>
                        <p>Confianza: <strong>{confidence_pct}%</strong></p>
                        <p>Texto analizado: <i>"{input_text[:50]}..."</i></p>
                        <p>Idioma: <strong>{'Inglés (traducido)' if language == 'Español' else 'Inglés'}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    st.progress(confidence)

st.markdown('<div class="content-box">', unsafe_allow_html=True)