import streamlit as st
from googletrans import Translator
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re

# Configuración de la página con fondo personalizado
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

# Aplicar CSS personalizado para el fondo y diseño
st.markdown("""
<style>
    /* Fondo animado */
    .stApp {
        background-image: url("https://64.media.tumblr.com/817c19affd93dc7dc145364acbb10331/8e4bb3b18c84e15f-60/s1280x1920/5cdcb9e6cb7edc05ab6994b12132f590033e7c0b.gifv");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Contenedor principal centrado */
    .main-container {
        max-width: 1200px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.85);
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Ajustes para los componentes de Streamlit */
    .stTextInput, .stTextArea, .stSelectbox, .stButton {
        margin-bottom: 1rem;
    }
    
    /* Títulos con mejor contraste */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Texto general */
    p, label, div {
        color: white !important;
    }
    
    /* Ajustar columnas */
    .stColumns {
        margin-bottom: 1rem;
    }
    
    /* Personalizar el botón */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    /* Personalizar el área de texto */
    .stTextArea>div>div>textarea {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Carga de Modelo y Tokenizer (con caché) ---
@st.cache_resource
def load_resources():
    # Paths relativos a la carpeta 'model'
    model_path = 'jgekko/sentio-app/main/model/modelSENTIO.h5'
    tokenizer_path = 'jgekko/sentio-app/main/model/tokenizer.pkl'
    
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_resources()
    if not model:
        raise ValueError("El modelo no se cargó correctamente.")
    if not tokenizer or not hasattr(tokenizer, "word_index"):
        raise ValueError("El tokenizer no se cargó correctamente.")
except Exception as e:
    st.error(f"Error cargando recursos: {str(e)}")
    st.stop()

# --- Funciones de Preprocesamiento ---
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

# --- Función de Traducción ---
def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Error en traducción: {e}")
        return text

# --- Función de Predicción ---
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

# --- Interfaz de Usuario ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🔍 Sentio - Análisis de Sentimientos")

# Instrucciones
st.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en español/inglés.
2. Selecciona el idioma del texto.
3. Haz clic en "Analizar Sentimiento".
""", unsafe_allow_html=True)

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

# Cerrar contenedor principal
st.markdown('</div>', unsafe_allow_html=True)