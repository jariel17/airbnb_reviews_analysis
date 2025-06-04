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
import spacy

nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

nlp_spacy = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def get_custom_stopwords():
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    stop_words.discard("no")
    stop_words.add("'")
    stop_words.add("´")
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
    
    # 1.4 Eliminar puntuación - se puede mejorar
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

def lemmatize_text_spacy_batch(text_series):
    lemmas = []
    total = len(text_series)
    # Definimos tamaño de chunk en 50000 para repartir en varias llamadas a pipe
    chunk_size = 50000
    for i in range(0, total, chunk_size):
        sub_list = text_series.iloc[i : i + chunk_size].tolist()
        docs = nlp_spacy.pipe(sub_list, batch_size=len(sub_list), n_process=1)
        lemmas.extend(
            " ".join([token.lemma_ for token in doc if token.lemma_ != "-PRON-"])
            for doc in docs
        )
    return lemmas


modelo = fasttext.load_model("lid.176.bin")

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in custom_stopwords]
    return " ".join(filtered)

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
