# Importación de librerías necesarias
import streamlit as st
from googletrans import Translator  # Para traducción de texto
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import re  # Para limpieza de texto
import urllib.request  # Para descargar los modelos
import os

# ==============================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================

st.set_page_config(
    page_title="Sentio - Análisis de Sentimientos",  # Título 
    layout="wide"  # Diseño amplio
)

# ==============================================
# ESTILO CSS PERSONALIZADO
# ==============================================

st.markdown("""
<style>
    /* Fondo animado para toda la aplicación */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://64.media.tumblr.com/817c19affd93dc7dc145364acbb10331/8e4bb3b18c84e15f-60/s1280x1920/5cdcb9e6cb7edc05ab6994b12132f590033e7c0b.gifv");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Contenedor principal transparente */
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Caja de contenido principal */
    .content-box {
        background-color: rgba(0, 0, 0, 0.85);  # Fondo negro semitransparente
        border-radius: 15px;  # Bordes redondeados
        padding: 6rem;  # Espaciado interno
        margin: 0 auto;  # Centrado
        max-width: 1200px;  # Ancho máximo
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);  # Sombra
        border: 1px solid rgba(255, 255, 255, 0.1);  # Borde sutil
    }

    /* Estilos para texto dentro del content-box */
    .content-box h1, 
    .content-box h2, 
    .content-box h3, 
    .content-box h4, 
    .content-box h5, 
    .content-box h6,
    .content-box p,
    .content-box label,
    .content-box div {
        color: white !important;  # Texto blanco
    }

    /* Personalización de componentes de Streamlit */
    .content-box .stTextInput input, 
    .content-box .stTextArea textarea {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    .content-box .stSelectbox select {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
    }
    
    /* Estilo para botones */
    .content-box .stButton>button {
        background-color: #4CAF50 !important;  # Verde
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }

    /* Estilo para la caja de resultados */
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

# ==============================================
# CONFIGURACIÓN DEL MODELO
# ==============================================

# Crear directorio 'model' si no existe
if not os.path.exists('model'):
    os.makedirs('model')

# Decorador para cachear recursos y mejorar rendimiento
@st.cache_resource
def load_resources():

    # Ruta de los archivos del modelo
    model_path = 'model/modelSENTIOfinal.keras'
    tokenizer_path = 'model/tokenizerSENTIOfinal.pkl'
    
    # Descargar archivos si no existen localmente
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_path)
    if not os.path.exists(tokenizer_path):
        urllib.request.urlretrieve(tokenizer_path)

    # Cargar modelo y tokenizer
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

# Cargar recursos al iniciar la aplicación
try:
    model, tokenizer = load_resources()
except Exception as e:
    st.error(f"Error cargando recursos: {str(e)}")
    st.stop()  # Detiene la ejecución si hay error

# ==============================================
# FUNCIONES DE PROCESAMIENTO
# ==============================================

# Limpia el texto eliminando caracteres especiales y convirtiendo a minúsculas.
def clean_text(texts):
    cleaned_texts = []
    for text in texts:
        text = text.lower() 
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    return cleaned_texts

# Preprocesa el texto como lo necesita el modelo para que pueda ser usado
def preprocess_text(texts, tokenizer, max_len=50):
    text_seq = tokenizer.texts_to_sequences(texts)  # Convertir texto a secuencias numéricas
    return pad_sequences(text_seq, maxlen=max_len, padding="post")  # Aplicar padding

# Traducir a ingles con Googletrans
def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Error en traducción: {e}")
        return text  # Devuelve texto original si hay error

# Realizar la prediccion
def predict_sentiment(text):
    try:
        text = [text]  # Convertir a lista para el procesamiento
        text = clean_text(text)  # Limpiar texto
        text_padded = preprocess_text(text, tokenizer)  # Preprocesar
        
        # Realizar predicción
        y_prob = model.predict(text_padded, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        
        # Mapear clases numéricas a etiquetas
        classes = ['Negative', 'Positive', 'Neutral']
        pred_class = classes[y_pred[0]]
        pred_prob = float(y_prob[0][y_pred[0]])
        
        # Mapeo de emojis y descripciones
        emoji_map = {
            'Negative': '😠 Negativo (tristeza/ira/desesperacion)',
            'Positive': '😊 Positivo (alegria/amor/entusiasmo)', 
            'Neutral': '😐 Neutral'
        }
        
        return emoji_map[pred_class], pred_prob
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None, None

# ==============================================
# INTERFAZ DE USUARIO
# ==============================================

# Título
st.title("🔍 Sentio - Análisis de Sentimientos")
    
# Instrucciones
st.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en español/inglés.
2. Selecciona el idioma del texto.
3. Haz clic en "Analizar Sentimiento".
""")
    
# Columnas del layout
col1, col2 = st.columns([1, 1])
    
# Columna izquierda - Entrada de datos
with col1:
    st.header("📝 Ingresa tu texto")
    language = st.selectbox("Idioma:", ["Español", "English"])  
    user_input = st.text_area("Escribe aquí:", max_chars=50, height=100)  
    analyze_btn = st.button("Analizar Sentimiento", type="primary") 
    
# Columna derecha - Resultados
with col2:
    st.header("📊 Resultado")
        
    if analyze_btn:  
        if not user_input:  
            st.warning("⚠️ Por favor ingresa texto")
        else:
            with st.spinner("Analizando..."): 
                # Traducir texto si es necesario
                input_text = translate_to_english(user_input) if language == "Español" else user_input
                # Obtener predicción
                sentiment, confidence = predict_sentiment(input_text)
                
                if sentiment and confidence:
                    confidence_pct = round(confidence * 100, 2)  # Convertir a porcentaje
                        
                    if "Positivo" in sentiment:
                        bg_color = "#339b2f"
                    elif "Negativo" in sentiment:
                        bg_color = "#9b2f2f" 
                    else:
                        bg_color = "#9b912f" 
                            
                    # Mostrar resultados con formato HTML
                    st.markdown(f"""
                    <div class="result-box" style="background: {bg_color}">
                        <h3>Predicción:</h3>
                        <p style='font-size: 24px;'>{sentiment}</p>
                        <p>Confianza: <strong>{confidence_pct}%</strong></p>
                        <p>Texto analizado: <i>"{input_text[:50]}..."</i></p>
                        <p>Idioma: <strong>{'Inglés (traducido)' if language == 'Español' else 'Inglés'}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    # Mostrar barra de progreso para la confianza
                    st.progress(confidence)