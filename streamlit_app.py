import streamlit as st
from googletrans import Translator
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Configuración de la página
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

# --- Carga de Modelo y Tokenizer (con caché) ---
@st.cache_resource
def load_resources():
    # Paths relativos a la carpeta 'model'
    model_path = os.path.join('model', 'modeloSENTIO.h5')
    tokenizer_path = os.path.join('model', 'tokenizer.pkl')
    
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"Error cargando recursos: {str(e)}")
    st.stop()  # Detiene la app si hay error

# --- Función de Traducción ---
def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Error en traducción: {e}")
        return text  # Fallback: usa texto original

# --- Función de Predicción ---
def predict_sentiment(text, max_len=50):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]  # Suprime output de TensorFlow
    return prediction

# --- Interfaz de Usuario ---
st.title("🔍 Sentio - Análisis de Sentimientos")
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Ingresa tu texto")
    language = st.selectbox("Idioma:", ["Español", "English"])
    user_input = st.text_area("Escribe aquí (máx. 50 caracteres):", max_chars=50, height=150)
    analyze_btn = st.button("Analizar Sentimiento", type="primary")

with col2:
    st.header("📊 Resultado")
    
    if analyze_btn and user_input:
        with st.spinner("Analizando..."):
            # Preprocesamiento según idioma
            input_text = translate_to_english(user_input) if language == "Español" else user_input
            
            if language == "Español":
                st.sidebar.info(f"Texto original: '{user_input}'")
                st.sidebar.info(f"Texto traducido: '{input_text}'")
            
            # Predicción real
            confidence = predict_sentiment(input_text)
            sentiment_emoji = "😊" if confidence > 0.5 else "😠"
            sentiment_text = f"{'Positivo' if confidence > 0.5 else 'Negativo'} {sentiment_emoji}"
            confidence_pct = round(float(confidence) * 100, 2)
            
            # Mostrar resultados
            with st.container():
                st.markdown(f"""
                <div class="result-panel">
                    <h3>Predicción:</h3>
                    <p style='font-size: 24px;'><strong>{sentiment_text}</strong></p>
                    <p>Confianza: <strong>{confidence_pct}%</strong></p>
                    <p>Idioma analizado: <strong>{'Inglés (traducido)' if language == 'Español' else 'Inglés'}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(confidence))
                
    elif analyze_btn and not user_input:
        st.warning("⚠️ Por favor ingresa texto antes de analizar")

# Sidebar
st.sidebar.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en español/inglés
2. Selecciona el idioma del texto
3. Haz clic en "Analizar Sentimiento"
""")

# Estilos CSS
st.markdown("""
<style>
    .result-panel {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    .stProgress > div > div > div {
        background-color: #FF4B4B;  /* Color rojo Sentio */
    }
</style>
""", unsafe_allow_html=True)