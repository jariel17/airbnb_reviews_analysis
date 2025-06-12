import re
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from collections import Counter
import pandas as pd
import string
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk
import fasttext

nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

def get_custom_stopwords():
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    stop_words.discard("no")
    stop_words.add("'")
    stop_words.add("´")
    stop_words.add("airbnb")
    stop_words.add("host")
    stop_words.add("place")
    stop_words.add("stay")
    stop_words.add("room")
    stop_words.add("definitely")
    stop_words.add("apartment")
    stop_words.add("would")
    stop_words.add("location")
    stop_words.add("mexico")
    return stop_words

custom_stopwords = get_custom_stopwords()

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # 1.1 Convertir a minúsculas
    text = text.lower()
    
    # 1.2 Eliminar HTML
    text = re.sub(r"<.*?>", " ", text)
    
    # 1.3 Eliminar URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    # 1.4 Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 1.5 Eliminar números
    text = re.sub(r"\d+", "", text)
    
    return text

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def lemmatize_text(text):
    tokens = word_tokenize(text)
    # Etiquetar POS
    pos_tags = pos_tag(tokens)
    lemmatized_tokens = []
    for token, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, wordnet_pos)
        lemmatized_tokens.append(lemma)
    return " ".join(lemmatized_tokens)


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in custom_stopwords]
    return " ".join(filtered)

def is_single_word(text):
        words = text.split()  # Divide el texto en palabras
        return len(words) == 1 

modelo = fasttext.load_model("lid.176.bin")

def detect_language(texto):
    try:
        texto = str(texto).strip()
        if texto == "":
            return "unknown"

        # limpieza mínima para mejorar precisión (quitar URLs y caracteres no alfanuméricos)
        limpio = re.sub(r"http\S+|www\S+", " ", texto)
        limpio = re.sub(r"<.*?>", " ", limpio)
        limpio = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]", " ", limpio)
        limpio = " ".join(limpio.split())
        if limpio == "":
            return "unknown"

        etiqueta_lista, prob_lista = modelo.predict([limpio])
        etiqueta = etiqueta_lista[0][0]
        idioma = etiqueta.replace("__label__", "")
        return idioma
    except Exception:
        return "unknown"
