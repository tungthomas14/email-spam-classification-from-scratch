const messageInput = document.querySelector("#messageInput");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");
const classifyButton = document.querySelector("#classifyButton");
const resultBadge = document.querySelector("#resultBadge");
const probabilityValue = document.querySelector("#probabilityValue");
const probabilityBar = document.querySelector("#probabilityBar");
const tokenStrip = document.querySelector("#tokenStrip");
const modelStatus = document.querySelector("#modelStatus");

const accuracyMetric = document.querySelector("#accuracyMetric");
const precisionMetric = document.querySelector("#precisionMetric");
const recallMetric = document.querySelector("#recallMetric");
const f1Metric = document.querySelector("#f1Metric");

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function setMetrics(metadata) {
  modelStatus.textContent = `${metadata.vocabulary_size} words`;
  accuracyMetric.textContent = formatPercent(metadata.test_accuracy);
  precisionMetric.textContent = formatPercent(metadata.test_precision);
  recallMetric.textContent = formatPercent(metadata.test_recall);
  f1Metric.textContent = formatPercent(metadata.test_f1_score);
}

function setResult(data) {
  const probability = data.probability;
  const percent = probability * 100;
  const isSpam = data.label === "spam";

  resultBadge.textContent = data.label.toUpperCase();
  resultBadge.classList.toggle("spam", isSpam);
  resultBadge.classList.toggle("ham", !isSpam);
  probabilityValue.textContent = formatPercent(probability);
  probabilityValue.classList.toggle("spam", isSpam);
  probabilityValue.classList.toggle("ham", !isSpam);
  probabilityBar.style.width = `${percent}%`;
  probabilityBar.style.backgroundColor = isSpam ? "var(--warn)" : "var(--accent)";

  tokenStrip.replaceChildren(
    ...data.tokens.slice(0, 28).map((token) => {
      const element = document.createElement("span");
      element.className = "token";
      element.textContent = token;
      return element;
    }),
  );
}

async function loadMetadata() {
  const response = await fetch("/api/meta");
  const metadata = await response.json();
  setMetrics(metadata);
}

async function classifyMessage() {
  const message = messageInput.value.trim();

  if (!message) {
    messageInput.focus();
    return;
  }

  classifyButton.disabled = true;
  classifyButton.textContent = "Analyzing";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        threshold: Number(thresholdInput.value),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Request failed");
    }

    setResult(data);
  } finally {
    classifyButton.disabled = false;
    classifyButton.textContent = "Analyze";
  }
}

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
});

classifyButton.addEventListener("click", classifyMessage);

messageInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    classifyMessage();
  }
});

loadMetadata();
