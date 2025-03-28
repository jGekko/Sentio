import streamlit as st
from googletrans import Translator
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re

# Configuración de la página
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

# CSS personalizado para el fondo y diseño
st.markdown("""
<style>
    /* Fondo animado fijo */
    .stApp {
        background-image: url("https://64.media.tumblr.com/817c19affd93dc7dc145364acbb10331/8e4bb3b18c84e15f-60/s1280x1920/5cdcb9e6cb7edc05ab6994b12132f590033e7c0b.gifv");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Contenedor principal que envuelve TODO el contenido */
    .main-content-wrapper {
        max-width: 1200px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.85);
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Asegurar que el contenido de Streamlit sea transparente */
    .stApp > div {
        background-color: transparent !important;
    }
    
    /* Estilos para el texto */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    p, label, div:not(.stAlert) {
        color: white !important;
    }
    
    /* Personalización de componentes */
    .stTextInput input, .stTextArea textarea {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .stSelectbox select {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    
    /* Ajustes para las columnas */
    .stColumns {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Código de carga del modelo y funciones (igual que antes) ---
@st.cache_resource
def load_resources():
    model_path = 'jgekko/sentio-app/main/model/modelSENTIO.h5'
    tokenizer_path = 'jgekko/sentio-app/main/model/tokenizer.pkl'
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# (Mantén aquí todas tus funciones igual: clean_text, preprocess_text, translate_to_english, predict_sentiment)

# --- Interfaz de usuario dentro del contenedor principal ---
with st.container():
    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
    
    # Título principal
    st.title("🔍 Sentio - Análisis de Sentimientos")
    
    # Instrucciones
    st.markdown("""
    ### ℹ️ Instrucciones:
    1. Escribe texto en español/inglés.
    2. Selecciona el idioma del texto.
    3. Haz clic en "Analizar Sentimiento".
    """)
    
    # Columnas para entrada y resultados
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
                            sentiment_color = "#D4EDDA"
                            text_color = "#155724"
                        elif "Negativo" in sentiment:
                            sentiment_color = "#F8D7DA"
                            text_color = "#721C24"
                        else:
                            sentiment_color = "#FFF3CD"
                            text_color = "#856404"
                            
                        st.markdown(f"""
                        <div style="
                            padding: 20px;
                            border-radius: 10px;
                            background: {sentiment_color};
                            color: {text_color};
                            margin-top: 20px;
                            font-weight: bold;
                        ">
                            <h3>Predicción:</h3>
                            <p style='font-size: 24px;'>{sentiment}</p>
                            <p>Confianza: <strong>{confidence_pct}%</strong></p>
                            <p>Texto analizado: <i>"{input_text[:50]}..."</i></p>
                            <p>Idioma: <strong>{'Inglés (traducido)' if language == 'Español' else 'Inglés'}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(confidence)
    
    # Cierre del contenedor principal
    st.markdown('</div>', unsafe_allow_html=True)