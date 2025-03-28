import re
from keras.preprocessing.sequence import pad_sequences

def clean_text(texts):
    """
    Limpia el texto (debe coincidir con tu preprocesamiento original)
    """
    cleaned_texts = []
    for text in texts:
        # Aplica las mismas transformaciones que en tu notebook
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        cleaned_texts.append(text)
    return cleaned_texts

def preprocess_text(texts, tokenizer, max_len=50):
    """
    Tokeniza y aplica padding (igual que en tu notebook)
    """
    text_seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(text_seq, maxlen=max_len, padding="post")