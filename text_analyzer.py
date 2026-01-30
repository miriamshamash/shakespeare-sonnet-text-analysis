import argparse
from pathlib import Path
import glob
import os
import re
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_top_k(kv_dict: Dict[str, float], k: int = 20) -> List[Tuple[str, float]]:
    """
    Returns the top 'k' key-value pairs from a dictionary, based on their values.
    """
    # Sort by value (highest first) and take top k
    top_k = sorted(kv_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_k


def sort_dictionary_by_value(
    dict_in: Dict[str, float], direction: str = "descending"
) -> List[Tuple[str, float]]:
    """
    Sort a dictionary of key-value pairs by their values.
    """
    # Sort ascending by default
    sort_dict = sorted(dict_in.items(), key=lambda x: x[1])

    # Reverse the order if the direction is 'descending'
    if direction.lower() == "descending":
        sort_dict = list(reversed(sort_dict))

    return sort_dict


def strip_non_ascii(string):
    """Returns the string without non ASCII characters"""
    stripped = (c for c in string if 0 < ord(c) < 127)
    return "".join(stripped)


def clean_text(s):
    """Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana'"""
    s = strip_non_ascii(s)
    s = re.sub("[^a-z A-Z]", "", s)
    s = s.replace(" n ", " ")
    return s


def clean_corpus(corpus):
    """Run clean_text() on each sonnet in the corpus"""
    for key in corpus.keys():
        corpus[key] = clean_text(corpus[key]).split()
    return corpus


def read_sonnets(fin):
    """
    Reads and cleans a directory of TXT files or a single TXT file.

    :param fin: fin can be a directory path containing TXT files to process or a single file path
    :return: dict mapping filename stem -> cleaned text string
    """
    if Path(fin).is_file():
        f_sonnets = [fin]
    elif Path(fin).is_dir():
        f_sonnets = glob.glob(fin + os.sep + "*.txt")
    else:
        print("Filepath of sonnet not found!")
        return None

    sonnets = {}
    for f in f_sonnets:
        sonnet_id = Path(f).stem

        # Read the whole file (your original code only read one line)
        with open(f, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()

        # Clean the text
        sonnets[sonnet_id] = clean_text(text.replace("\n", " ").replace("\r", " "))

    return sonnets


def tf(document: List[str]) -> Dict[str, int]:
    """
    Calculate the term frequency (TF) for each word in a document.
    """
    document_tf: Dict[str, int] = {}

    for word in document:
        if word == "":
            continue
        word = word.lower()
        document_tf[word] = document_tf.get(word, 0) + 1

    return document_tf


def idf(corpus: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate the inverted document frequency (IDF) for each word in a corpus.
    IDF(word) = log(N / df)
    """
    N = len(corpus)
    if N == 0:
        return {}

    # df counts how many documents each word appears in (at least once)
    df: Dict[str, int] = {}

    for doc_words in corpus.values():
        unique_words = set(w.lower() for w in doc_words if w != "")
        for w in unique_words:
            df[w] = df.get(w, 0) + 1

    corpus_idf: Dict[str, float] = {}
    for w, doc_count in df.items():
        # Avoid division by zero (shouldn't happen, but safe)
        if doc_count == 0:
            continue
        corpus_idf[w] = math.log(N / doc_count)

    return corpus_idf


def tf_idf(corpus_idf: Dict[str, float], sonnet_tf: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate TF-IDF scores for each word in a document using a precomputed IDF dictionary.
    TF-IDF(word) = TF(word) * IDF(word)
    """
    corpus_tf_idf: Dict[str, float] = {}

    for w, tf_val in sonnet_tf.items():
        w_lower = w.lower()
        idf_val = corpus_idf.get(w_lower, 0.0)
        corpus_tf_idf[w_lower] = float(tf_val) * float(idf_val)

    return corpus_tf_idf


def cosine_sim(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Calculate the cosine similarity between two tf-idf vectors represented as dicts.
    """
    # Dot product over intersection of keys
    common_keys = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in common_keys)

    # Magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))

    # Avoid divide-by-zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Analysis through TFIDF computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./data/shakespeare_sonnets/1.txt",
        help="Input text file or files.",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        default="./data/shakespeare_sonnets/",
        help="Directory containing document collection (i.e., corpus)",
    )
    parser.add_argument(
        "--tfidf",
        help="Determine the TF IDF of a document w.r.t. a given corpus",
        action="store_true",
    )

    args = parser.parse_args()

    # return dictionary with keys corresponding to file names and values being the respective contents
    corpus = read_sonnets(args.corpus)
    if corpus is None:
        raise SystemExit("Could not read corpus.")

    # return corpus (dict) with each sonnet cleaned and tokenized for further processing
    corpus = clean_corpus(corpus)

    # assign 1.txt to variable sonnet to process and find its TF
    if "1" not in corpus:
        raise SystemExit("Sonnet '1' not found in corpus. Check your file names / paths.")

    sonnet1 = corpus["1"]

    # determine tf of sonnet
    sonnet1_tf = tf(sonnet1)

    # get sorted list and slice out top 20
    sonnet1_top20 = get_top_k(sonnet1_tf)
    print("\nSonnet 1 TF (Top 20):")
    print(sonnet1_top20)

    # TF of entire corpus
    flattened_corpus = [word for sonnet in corpus.values() for word in sonnet]
    corpus_tf = tf(flattened_corpus)
    corpus_top20 = get_top_k(corpus_tf)
    print("\nCorpus TF (Top 20):")
    print(corpus_top20)

    # IDF of corpus
    corpus_idf = idf(corpus)
    corpus_idf_top20 = get_top_k(corpus_idf)
    print("\nCorpus IDF (Top 20):")
    print(corpus_idf_top20)

    # TFIDF of Sonnet1 w.r.t. corpus
    sonnet1_tfidf = tf_idf(corpus_idf, sonnet1_tf)
    sonnet1_tfidf_top20 = get_top_k(sonnet1_tfidf)
    print("\nSonnet 1 TFIDF (Top 20):")
    print(sonnet1_tfidf_top20)

    # Example cosine similarity between Sonnet 1 and Sonnet 2 if it exists
    if "2" in corpus:
        sonnet2_tf = tf(corpus["2"])
        sonnet2_tfidf = tf_idf(corpus_idf, sonnet2_tf)
        sim_1_2 = cosine_sim(sonnet1_tfidf, sonnet2_tfidf)
        print("\nCosine similarity (Sonnet 1 vs Sonnet 2):")
        print(sim_1_2)
