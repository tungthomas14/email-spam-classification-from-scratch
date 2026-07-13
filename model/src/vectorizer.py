import numpy as np


def build_vocabulary(tokenized_texts):
    """Create a token-to-index dictionary from tokenized texts."""
    vocabulary = {}

    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


def texts_to_bow(tokenized_texts, vocabulary):
    """Convert tokenized texts into Bag-of-Words vectors."""
    bow_vectors = np.zeros((len(tokenized_texts), len(vocabulary)), dtype=np.float32)

    for row_index, tokens in enumerate(tokenized_texts):
        for token in tokens:
            if token in vocabulary:
                token_index = vocabulary[token]
                bow_vectors[row_index, token_index] += 1

    return bow_vectors
