import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function cleanText(text) {
  return text
    .toLowerCase()
    .replace(/[^a-zA-Z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  const cleanedText = cleanText(text);
  return cleanedText ? cleanedText.split(" ") : [];
}

function sigmoid(value) {
  const clipped = Math.max(-500, Math.min(500, value));
  return 1 / (1 + Math.exp(-clipped));
}

function predictSpamProbability(tokens, model) {
  let score = model.bias;

  for (const token of tokens) {
    const index = model.vocabulary[token];

    if (index !== undefined) {
      score += model.weights[index];
    }
  }

  return sigmoid(score);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [model, setModel] = useState(null);
  const [message, setMessage] = useState("Congratulations! You won a free prize. Call now.");
  const [threshold, setThreshold] = useState(0.55);
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch("/model.json")
      .then((response) => response.json())
      .then((payload) => {
        setModel(payload);
        setThreshold(payload.threshold);
      });
  }, []);

  const metadata = model?.metadata;

  const classifyMessage = () => {
    if (!model || !message.trim()) {
      return;
    }

    const tokens = tokenize(message);
    const probability = predictSpamProbability(tokens, model);
    const label = probability >= threshold ? "spam" : "ham";

    setResult({
      label,
      probability,
      tokens,
    });
  };

  const probabilityWidth = useMemo(() => {
    if (!result) {
      return "0%";
    }

    return `${result.probability * 100}%`;
  }, [result]);

  const isSpam = result?.label === "spam";

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="header-row">
          <div>
            <p className="eyebrow">Logistic Regression</p>
            <h1>Email Spam Classifier</h1>
          </div>
          <div className="status-pill">{metadata ? `${metadata.vocabulary_size} words` : "Loading model"}</div>
        </div>

        <div className="tool-grid">
          <section className="composer" aria-label="Message classifier">
            <label htmlFor="messageInput">Message</label>
            <textarea
              id="messageInput"
              spellCheck="false"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
            />

            <div className="control-row">
              <label htmlFor="thresholdInput">Threshold {Number(threshold).toFixed(2)}</label>
              <input
                id="thresholdInput"
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
            </div>

            <button type="button" onClick={classifyMessage} disabled={!model}>
              Analyze
            </button>
          </section>

          <section className="result-panel" aria-label="Prediction result">
            <div className="result-topline">
              <p className="eyebrow">Prediction</p>
              <span className={`result-badge ${result ? result.label : ""}`}>{result ? result.label.toUpperCase() : "Waiting"}</span>
            </div>
            <div className={`probability ${result ? result.label : ""}`}>{result ? formatPercent(result.probability) : "--"}</div>
            <div className="bar-track" aria-hidden="true">
              <div className="bar-fill" style={{ width: probabilityWidth, backgroundColor: isSpam ? "var(--warn)" : "var(--accent)" }} />
            </div>
            <div className="token-strip">
              {(result?.tokens || []).slice(0, 28).map((token, index) => (
                <span className="token" key={`${token}-${index}`}>
                  {token}
                </span>
              ))}
            </div>
          </section>
        </div>

        <section className="metrics-panel" aria-label="Model metrics">
          <div className="metric">
            <span>Accuracy</span>
            <strong>{metadata ? formatPercent(metadata.test_accuracy) : "--"}</strong>
          </div>
          <div className="metric">
            <span>Precision</span>
            <strong>{metadata ? formatPercent(metadata.test_precision) : "--"}</strong>
          </div>
          <div className="metric">
            <span>Recall</span>
            <strong>{metadata ? formatPercent(metadata.test_recall) : "--"}</strong>
          </div>
          <div className="metric">
            <span>F1</span>
            <strong>{metadata ? formatPercent(metadata.test_f1_score) : "--"}</strong>
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
