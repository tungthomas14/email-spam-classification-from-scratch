import csv
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "spam.csv"

LABEL_TO_ID = {
    "ham": 0,
    "spam": 1,
}


def read_data(file_path=DEFAULT_DATA_PATH):
    """Read the SMS spam dataset as two lists: labels and texts."""
    labels = []
    texts = []

    with Path(file_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for line_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(
                    f"Invalid row at line {line_number}: expected 2 columns, got {len(row)}"
                )

            label, text = row
            labels.append(label)
            texts.append(text)

    return texts, labels


def encode_labels(labels):
    """Convert labels from ham/spam to 0/1."""
    encoded_labels = []

    for label in labels:
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unknown label: {label}")

        encoded_labels.append(LABEL_TO_ID[label])

    return encoded_labels


def lowercase_texts(texts):
    """Convert all texts to lowercase."""
    lowercased_texts = []

    for text in texts:
        lowercased_texts.append(text.lower())

    return lowercased_texts


def remove_punctuation_texts(texts):
    """Remove punctuation and special characters from all texts."""
    cleaned_texts = []

    for text in texts:
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        cleaned_texts.append(cleaned_text)

    return cleaned_texts


def tokenize_texts(texts):
    """Split all texts into word tokens."""
    tokenized_texts = []

    for text in texts:
        tokens = text.split()
        tokenized_texts.append(tokens)

    return tokenized_texts


if __name__ == "__main__":
    texts, labels = read_data()
    y = encode_labels(labels)
    lowercased_texts = lowercase_texts(texts)
    cleaned_texts = remove_punctuation_texts(lowercased_texts)
    tokenized_texts = tokenize_texts(cleaned_texts)

    print(f"Number of samples: {len(texts)}")
    print(f"First text: {texts[0]}")
    print(f"First lowercased text: {lowercased_texts[0]}")
    print(f"First cleaned text: {cleaned_texts[0]}")
    print(f"First tokens: {tokenized_texts[0]}")
    print(f"First label: {labels[0]} -> {y[0]}")
