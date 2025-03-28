import streamlit as st
from googletrans import Translator

# Configuración de la página
st.set_page_config(page_title="Sentio - Análisis de Sentimientos", layout="wide")

# Estilos CSS personalizados
st.markdown("""
<style>
    .stTextInput>div>div>input {
        max-width: 400px;
    }
    .stSelectbox>div>div>select {
        max-width: 200px;
    }
    .result-panel {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Título de la app
st.title("🔍 Sentio - Análisis de Sentimientos")

# Dividir en dos columnas
col1, col2 = st.columns([1, 1])

# Panel izquierdo: Entrada de texto
with col1:
    st.header("📝 Ingresa tu texto")
    
    # Selector de idioma
    language = st.selectbox("Idioma:", ["Español", "English"])
    
    # Cuadro de texto con contador de caracteres
    user_input = st.text_area(
        "Escribe aquí (máx. 50 caracteres):", 
        max_chars=50,
        height=150,
        key="text_input"
    )
    
    # Botón de análisis
    analyze_btn = st.button("Analizar Sentimiento", type="primary")

# Función de traducción
def translate_to_english(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='es', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Error en traducción: {e}")
        return text  # Si falla, envía el texto original

# Panel derecho: Resultados (modifica esta parte)
with col2:
    st.header("📊 Resultado")
    
    if analyze_btn and user_input:
        with st.spinner("Analizando..."):
            # --- TRADUCCIÓN SI ES ESPAÑOL ---
            input_text = user_input
            if language == "Español":
                input_text = translate_to_english(user_input)
                st.sidebar.info(f"Texto traducido: '{input_text}'")  # Opcional: mostrar traducción
            
            # --- SIMULACIÓN DE MODELO (REMPLAZAR CON TU MODELO REAL) ---
            # (Ahora usa input_text en lugar de user_input)
            if language == "Español":
                sentiment = "Positivo 😊" if len(input_text) % 2 == 0 else "Negativo 😠"
            else:
                sentiment = "Positive 😊" if len(user_input) % 2 == 0 else "Negative 😠"
                confidence = round(abs(len(user_input)/50 * 100), 2)
            # --- FIN SIMULACIÓN ---
            
            # Mostrar resultados
            with st.container():
                st.markdown(f"""
                <div class="result-panel">
                    <h3>Predicción:</h3>
                    <p style='font-size: 24px;'><strong>{sentiment}</strong></p>
                    <p>Confianza: <strong>{confidence}%</strong></p>
                    <p>Idioma seleccionado: <strong>{language}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Ejemplo de barra de progreso
                st.progress(confidence / 100)
                
    elif analyze_btn and not user_input:
        st.warning("⚠️ Por favor ingresa texto antes de analizar")
    else:
        st.info("👈 Escribe texto y haz clic en 'Analizar Sentimiento'")

# Notas adicionales
st.sidebar.markdown("""
### ℹ️ Instrucciones:
1. Escribe texto en el cuadro (máx. 50 caracteres)
2. Selecciona el idioma
3. Haz clic en "Analizar Sentimiento"
""")