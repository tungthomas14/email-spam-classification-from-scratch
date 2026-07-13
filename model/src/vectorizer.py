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
    bow_vectors = []

    for tokens in tokenized_texts:
        vector = [0] * len(vocabulary)

        for token in tokens:
            if token in vocabulary:
                token_index = vocabulary[token]
                vector[token_index] += 1

        bow_vectors.append(vector)

    return bow_vectors
