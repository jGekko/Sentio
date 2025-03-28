import streamlit as st
from googletrans import Translator
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re

# Configuración de la página
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

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
    """Limpia el texto igual que en tu notebook"""
    cleaned_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    return cleaned_texts

def preprocess_text(texts, tokenizer, max_len=50):
    """Tokeniza y aplica padding igual que en tu notebook"""
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
        return text  # Fallback: usa texto original

# --- Función de Predicción ---
def predict_sentiment(text):
    """Devuelve (clase_predicha, probabilidad) como en tu notebook"""
    try:
        # Preprocesamiento idéntico al notebook
        text = [text]
        text = clean_text(text)
        text_padded = preprocess_text(text, tokenizer)
        
        # Predicción
        y_prob = model.predict(text_padded, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        classes = ['Negative', 'Positive', 'Neutral']
        pred_class = classes[y_pred[0]]
        pred_prob = float(y_prob[0][y_pred[0]])
        
        # Mapeo a emojis para Streamlit
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
st.title("🔍 Sentio - Análisis de Sentimientos")

# Mostrar instrucciones debajo del título
st.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en español/inglés en el cuadro de texto
2. Selecciona el idioma del texto
3. Haz clic en "Analizar Sentimiento" para ver los resultados
""")

# Crear espacio entre instrucciones y campos de entrada
st.write("")

# Sección de entrada y resultados (sin usar columnas)
with st.container():
    st.header("📝 Ingresa tu texto")
    language = st.selectbox("Idioma:", ["Español", "English"])
    user_input = st.text_area("Escribe tu texto aquí:", max_chars=200, height=100, 
                            placeholder="Ejemplo: Estoy muy contento con este producto...")
    analyze_btn = st.button("Analizar Sentimiento", type="primary", use_container_width=True)

# Espacio antes de los resultados
st.write("")

# Mostrar resultados cuando se hace clic en el botón
if analyze_btn:
    if not user_input:
        st.warning("⚠️ Por favor ingresa texto antes de analizar")
    else:
        with st.spinner("Analizando el sentimiento..."):
            # Traducción si es necesario
            input_text = translate_to_english(user_input) if language == "Español" else user_input
            
            # Predicción
            sentiment, confidence = predict_sentiment(input_text)
            
            if sentiment and confidence:
                confidence_pct = round(confidence * 100, 2)
                
                # Determinar estilo según el sentimiento
                if "Positivo" in sentiment:
                    bg_color = "#D4EDDA"  # Verde claro
                    border_color = "#C3E6CB"
                    text_color = "#155724"
                    emoji = "😊"
                elif "Negativo" in sentiment:
                    bg_color = "#F8D7DA"  # Rojo claro
                    border_color = "#F5C6CB"
                    text_color = "#721C24"
                    emoji = "😠"
                else:
                    bg_color = "#FFF3CD"  # Amarillo claro
                    border_color = "#FFEEBA"
                    text_color = "#856404"
                    emoji = "😐"
                
                # Mostrar resultados con formato mejorado
                st.markdown(f"""
                <div style="
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    background: {bg_color};
                    border-left: 5px solid {border_color};
                    margin: 1rem 0;
                    color: {text_color};
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 2rem; margin-right: 1rem;">{emoji}</span>
                        <h3 style="margin: 0; color: {text_color};">Resultado del Análisis</h3>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Sentimiento detectado</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{sentiment}</p>
                        </div>
                        
                        <div>
                            <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Nivel de confianza</p>
                            <p style="margin: 0; font-size: 1.5rem; font-weight: bold;">{confidence_pct}%</p>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1rem;">
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Texto analizado</p>
                        <p style="margin: 0; font-style: italic;">"{input_text[:70]}{'...' if len(input_text) > 70 else ''}"</p>
                    </div>
                    
                    <div style="margin-top: 0.5rem;">
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">Idioma de análisis</p>
                        <p style="margin: 0;">{'Inglés (traducido automáticamente)' if language == 'Español' else 'Inglés (original)'}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Barra de progreso con estilo
                st.progress(
                    value=confidence,
                    text=f"Nivel de confianza: {confidence_pct}%"
                )

# Estilos CSS adicionales
st.markdown("""
<style>
    /* Estilo para el botón principal */
    .stButton>button {
        width: 100%;
        padding: 0.5rem;
        font-weight: bold;
    }
    
    /* Mejorar el área de texto */
    .stTextArea textarea {
        min-height: 100px;
    }
    
    /* Espaciado general */
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)