from collections import Counter
import pandas as pd
import string
import seaborn as sns
import matplotlib.pyplot as plt
from  wordcloud import WordCloud

def plot_top_words(df, col, top_n=20, ax=None, title=""):
    # Concatenar todos los textos y tokenizar
    all_tokens = " ".join(df[col].dropna().tolist()).split()
    # Contar frecuencia
    counter = Counter(all_tokens)
    top = counter.most_common(top_n)
    words, freqs = zip(*top)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(freqs), y=list(words), palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Palabra")
    return ax

def compare_top_words(df, col1, col2, top_n=20):
    counter1 = Counter(" ".join(df[col1].dropna().tolist()).split())
    counter2 = Counter(" ".join(df[col2].dropna().tolist()).split())
    top1 = counter1.most_common(top_n)
    top2 = counter2.most_common(top_n)

    df_plot = pd.DataFrame([
        {"palabra": w, "frecuencia": f, "método": "spaCy"}
        for w, f in top1
    ] + [
        {"palabra": w, "frecuencia": f, "método": "NLTK"}
        for w, f in top2
    ])

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    sns.barplot(
        data=df_plot[df_plot["método"]=="spaCy"],
        x="frecuencia", y="palabra", palette="Blues_d", ax=ax1,
        order=[w for w, _ in top1]
    )
    ax1.set_title(f"Top {top_n} palabras ({col1})")
    ax1.set_xlabel("Frecuencia")
    ax1.set_ylabel("Palabra")

    sns.barplot(
        data=df_plot[df_plot["método"]=="NLTK"],
        x="frecuencia", y="palabra", palette="Greens_d", ax=ax2,
        order=[w for w, _ in top2]
    )
    ax2.set_title(f"Top {top_n} palabras ({col2})")
    ax2.set_xlabel("Frecuencia")
    ax2.set_ylabel("")

    plt.tight_layout()
    plt.show()


def plot_wordcloud_from_matrix(tfidf_matrix, feature_names, top_n=100, title="WordCloud", ax=None):

    import numpy as np
    sums = np.asarray(tfidf_matrix.sum(axis=0)).ravel()

    idx_sorted = np.argsort(sums)[::-1]
    top_idx = idx_sorted[:top_n]

    freqs = { feature_names[i]: float(sums[i]) for i in top_idx if sums[i] > 0 }

    wc = WordCloud(width=800,
                   height=400,
                   background_color="white")\
         .generate_from_frequencies(freqs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")