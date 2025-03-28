from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_loader import load_model, load_tokenizer

def predict_sentiment(text: str, max_len: int = 50) -> tuple:
    """Devuelve (sentimiento, confianza)"""
    model = load_model()
    tokenizer = load_tokenizer()
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    prediction = model.predict(padded)[0][0]
    sentiment = "Positivo 😊" if prediction > 0.5 else "Negativo 😠"
    
    return sentiment, float(prediction)