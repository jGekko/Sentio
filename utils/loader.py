import tensorflow as tf
import pickle
from functools import lru_cache

@lru_cache(maxsize=None)  # Cache en memoria para evitar recargas
def load_model():
    return tf.keras.models.load_model('modelo_sentio.h5')

@lru_cache(maxsize=None)
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)