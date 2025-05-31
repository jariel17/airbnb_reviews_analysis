import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from collections import Counter
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
import fasttext
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1.1 Convertir a minúsculas
    text = text.lower()
    
    # 1.2 Eliminar HTML
    text = re.sub(r"<.*?>", " ", text)
    
    # 1.3 Eliminar URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    # 1.4 Eliminar puntuación - se puede mejorar
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 1.5 Eliminar números
    text = re.sub(r"\d+", "", text)
    
    # 1.6 Eliminar stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    
    return text


modelo = fasttext.load_model("lid.176.bin")

def detect_language(texto):
    """
    Detecta el idioma de un texto usando FastText. 
    Retorna el código ISO 639-1 (por ejemplo, "en", "es").
    Si el texto está vacío o ocurre un error, retorna "unknown".
    """
    try:
        texto = str(texto).strip()
        if texto == "":
            return "unknown"

        # Opcional: limpieza mínima para mejorar precisión (quitar URLs y caracteres no alfanuméricos)
        limpio = re.sub(r"http\S+|www\S+", " ", texto)
        limpio = re.sub(r"<.*?>", " ", limpio)
        limpio = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]", " ", limpio)
        limpio = " ".join(limpio.split())
        if limpio == "":
            return "unknown"

        # FastText devuelve ([['__label__xx']], [[probabilidad]])
        etiqueta_lista, prob_lista = modelo.predict([limpio])
        etiqueta = etiqueta_lista[0][0]    # e.g., "__label__en"
        idioma = etiqueta.replace("__label__", "")
        return idioma
    except Exception:
        return "unknown"
