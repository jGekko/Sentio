from loader import load_model, load_tokenizer
from preprocessor import clean_text, preprocess_text
import numpy as np

def predict_sentiment(text: str) -> tuple:
    """
    Devuelve (clase_predicha, probabilidad)
    Coincide exactamente con tu implementación del notebook
    """
    model = load_model()
    tokenizer = load_tokenizer()
    
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