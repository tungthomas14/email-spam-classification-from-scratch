import json
from pathlib import Path

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


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
MODEL_OUTPUT_PATH = ROOT_DIR / "public" / "model.json"
DEFAULT_THRESHOLD = 0.55


def preprocess_messages(messages):
    lowercased_texts = lowercase_texts(messages)
    cleaned_texts = remove_punctuation_texts(lowercased_texts)
    return tokenize_texts(cleaned_texts)


def main():
    texts, labels = read_data()
    y = encode_labels(labels)
    tokenized_texts = preprocess_messages(texts)

    tokenized_train, tokenized_test, y_train, y_test = train_test_split(
        tokenized_texts,
        y,
        test_size=0.2,
        seed=42,
    )

    vocabulary = build_vocabulary(tokenized_train)
    X_train = texts_to_bow(tokenized_train, vocabulary)
    X_test = texts_to_bow(tokenized_test, vocabulary)

    model = LogisticRegression(learning_rate=0.1, epochs=100, class_weight="balanced")
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test, threshold=DEFAULT_THRESHOLD)

    payload = {
        "threshold": DEFAULT_THRESHOLD,
        "vocabulary": vocabulary,
        "weights": model.weights.astype(float).tolist(),
        "bias": float(model.bias),
        "metadata": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "vocabulary_size": len(vocabulary),
            "initial_loss": float(model.loss_history[0]),
            "final_loss": float(model.loss_history[-1]),
            "test_accuracy": float(accuracy(y_test, test_predictions)),
            "test_precision": float(precision(y_test, test_predictions)),
            "test_recall": float(recall(y_test, test_predictions)),
            "test_f1_score": float(f1_score(y_test, test_predictions)),
            "test_confusion_matrix": confusion_matrix(y_test, test_predictions),
        },
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
