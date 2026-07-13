from logistic_regression import LogisticRegression
from metrics import accuracy, confusion_matrix, f1_score, precision, recall
from preprocessing import (
    encode_labels,
    lowercase_texts,
    read_data,
    remove_punctuation_texts,
    tokenize_texts,
    train_test_split,
)
from vectorizer import build_vocabulary, texts_to_bow


def main():
    texts, labels = read_data()
    y = encode_labels(labels)

    lowercased_texts = lowercase_texts(texts)
    cleaned_texts = remove_punctuation_texts(lowercased_texts)
    tokenized_texts = tokenize_texts(cleaned_texts)

    tokenized_train, tokenized_test, y_train, y_test = train_test_split(
        tokenized_texts,
        y,
        test_size=0.2,
        seed=42,
    )

    vocabulary = build_vocabulary(tokenized_train)
    X_train = texts_to_bow(tokenized_train, vocabulary)
    X_test = texts_to_bow(tokenized_test, vocabulary)

    model = LogisticRegression(learning_rate=0.1, epochs=100)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Initial loss: {model.loss_history[0]:.4f}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
    print(f"Train accuracy: {accuracy(y_train, train_predictions):.4f}")
    print(f"Test accuracy: {accuracy(y_test, test_predictions):.4f}")
    print(f"Test precision: {precision(y_test, test_predictions):.4f}")
    print(f"Test recall: {recall(y_test, test_predictions):.4f}")
    print(f"Test F1-score: {f1_score(y_test, test_predictions):.4f}")
    print(f"Test confusion matrix: {confusion_matrix(y_test, test_predictions)}")


if __name__ == "__main__":
    main()
