import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

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
WEB_DIR = PROJECT_DIR / "web"


def preprocess_messages(messages):
    lowercased_texts = lowercase_texts(messages)
    cleaned_texts = remove_punctuation_texts(lowercased_texts)
    return tokenize_texts(cleaned_texts)


def train_classifier():
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

    model = LogisticRegression(learning_rate=0.1, epochs=100)
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    metadata = {
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
    }

    return model, vocabulary, metadata


MODEL, VOCABULARY, METADATA = train_classifier()


class SpamClassifierHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/api/meta":
            self.send_json_headers()
            return

        file_path = self.resolve_static_path(parsed_path.path)

        if file_path is None:
            self.send_error(404)
            return

        self.send_static_headers(file_path)

    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/api/meta":
            self.send_json(METADATA)
            return

        file_path = self.resolve_static_path(parsed_path.path)

        if file_path is None:
            self.send_error(404)
            return

        self.send_static_file(file_path)

    def do_POST(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path != "/api/predict":
            self.send_error(404)
            return

        try:
            payload = self.read_json_body()
            message = payload.get("message", "")
            threshold = float(payload.get("threshold", 0.5))

            if not message.strip():
                self.send_json({"error": "Message is required"}, status=400)
                return

            tokenized_message = preprocess_messages([message])
            vector = texts_to_bow(tokenized_message, VOCABULARY)
            probability = float(MODEL.predict_proba(vector)[0])
            prediction = 1 if probability >= threshold else 0

            self.send_json(
                {
                    "label": "spam" if prediction == 1 else "ham",
                    "probability": probability,
                    "threshold": threshold,
                    "tokens": tokenized_message[0],
                }
            )
        except (json.JSONDecodeError, ValueError) as error:
            self.send_json({"error": str(error)}, status=400)

    def read_json_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def resolve_static_path(self, request_path):
        if request_path == "/":
            request_path = "/index.html"

        relative_path = request_path.lstrip("/")
        file_path = (WEB_DIR / relative_path).resolve()

        if not file_path.is_file() or WEB_DIR.resolve() not in file_path.parents:
            return None

        return file_path

    def send_static_file(self, file_path):
        self.send_static_headers(file_path)
        self.wfile.write(file_path.read_bytes())

    def send_static_headers(self, file_path):
        content_type = self.get_content_type(file_path)
        content = file_path.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()

    def send_json(self, payload, status=200):
        content = json.dumps(payload).encode("utf-8")

        self.send_json_headers(status=status, content_length=len(content))
        self.wfile.write(content)

    def send_json_headers(self, status=200, content_length=0):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def get_content_type(self, file_path):
        suffix = file_path.suffix

        if suffix == ".html":
            return "text/html; charset=utf-8"
        if suffix == ".css":
            return "text/css; charset=utf-8"
        if suffix == ".js":
            return "application/javascript; charset=utf-8"
        if suffix == ".svg":
            return "image/svg+xml"

        return "application/octet-stream"

    def log_message(self, format, *args):
        return


def main():
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), SpamClassifierHandler)
    print(f"Server running at http://{host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
