const API_BASE_URL = "https://human-skin.onrender.com";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const chooseBtn = document.getElementById("choose-btn");
const resetBtn = document.getElementById("reset-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const uploadInfo = document.getElementById("upload-info");
const statusText = document.getElementById("status-text");

const previewImage = document.getElementById("preview-image");
const heatmapImage = document.getElementById("heatmap-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const toggleHeatmapBtn = document.getElementById("toggle-heatmap-btn");

const progressBar = document.getElementById("progress-bar");

const resultsSection = document.getElementById("results-section");
const predLabelEl = document.getElementById("pred-label");
const predProbEl = document.getElementById("pred-prob");
const uncertaintyTag = document.getElementById("uncertainty-tag");
const topkList = document.getElementById("topk-list");
const metricRedness = document.getElementById("metric-redness");
const metricCyanosis = document.getElementById("metric-cyanosis");
const metricArea = document.getElementById("metric-area");
const qualityHint = document.getElementById("quality-hint");

const toastRoot = document.getElementById("toast-root");

let currentFile = null;
let currentInferenceId = null;
let currentHeatmapVisible = false;

const STATE = {
  IDLE: "IDLE",
  READY: "READY",
  PROCESSING: "PROCESSING",
  DONE: "DONE",
};

let appState = STATE.IDLE;

function setState(newState) {
  appState = newState;

  if (newState === STATE.IDLE) {
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    statusText.textContent = "No image selected.";
  } else if (newState === STATE.READY) {
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    statusText.textContent = "Image is ready to be analyzed.";
  } else if (newState === STATE.PROCESSING) {
    analyzeBtn.disabled = true;
    resetBtn.disabled = true;
    statusText.textContent = "Analyzing image…";
  } else if (newState === STATE.DONE) {
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    statusText.textContent = "Analysis completed.";
  }
}

function showToast(type, title, message, timeoutMs = 4000) {
  const toast = document.createElement("div");
  toast.className = "toast";

  if (type === "success") toast.classList.add("toast-success");
  else if (type === "error") toast.classList.add("toast-error");
  else if (type === "warning") toast.classList.add("toast-warning");

  const content = document.createElement("div");
  const titleEl = document.createElement("div");
  titleEl.className = "toast-title";
  titleEl.textContent = title;
  const msgEl = document.createElement("p");
  msgEl.className = "toast-message";
  msgEl.textContent = message;
  content.appendChild(titleEl);
  content.appendChild(msgEl);

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast-close";
  closeBtn.innerHTML = "×";
  closeBtn.addEventListener("click", () => toast.remove());

  toast.appendChild(content);
  toast.appendChild(closeBtn);

  toastRoot.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, timeoutMs);
}

function clearResults() {
  resultsSection.classList.add("hidden");
  predLabelEl.textContent = "—";
  predProbEl.textContent = "—";
  uncertaintyTag.textContent = "—";
  uncertaintyTag.className = "uncertainty-tag";
  topkList.innerHTML = "";
  metricRedness.textContent = "—";
  metricCyanosis.textContent = "—";
  metricArea.textContent = "—";
  qualityHint.textContent = "";
  toggleHeatmapBtn.classList.add("hidden");
  heatmapImage.classList.remove("visible");
  currentHeatmapVisible = false;
  toggleHeatmapBtn.textContent = "Show decision heatmap";
}

function clearPreview() {
  previewImage.src = "";
  heatmapImage.src = "";
  previewImage.classList.add("hidden");
  heatmapImage.classList.add("hidden");
  previewPlaceholder.classList.remove("hidden");
}

async function callCleanupIfNeeded() {
  if (!currentInferenceId) return;
  try {
    await fetch(`${API_BASE_URL}/cleanup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inference_id: currentInferenceId }),
    });
  } catch (err) {
    console.warn("Cleanup request failed:", err);
  } finally {
    currentInferenceId = null;
  }
}

function handleNewFile(file) {
  if (!file) return;

  const maxSizeBytes = 5 * 1024 * 1024;
  if (file.size > maxSizeBytes) {
    showToast(
      "error",
      "File too large",
      "The file is larger than 5MB. Please choose a smaller image."
    );
    return;
  }

  const allowedTypes = ["image/png", "image/jpeg"];
  if (!allowedTypes.includes(file.type)) {
    showToast(
      "warning",
      "Unsupported file type",
      "Please upload a PNG or JPG image."
    );
    return;
  }

  callCleanupIfNeeded();

  currentFile = file;
  clearResults();

  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.onload = () => {
    URL.revokeObjectURL(url);
  };

  previewImage.classList.remove("hidden");
  heatmapImage.classList.add("hidden");
  previewPlaceholder.classList.add("hidden");

  uploadInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(
    1
  )} KB)`;

  setState(STATE.READY);
}

function onDropzoneClick() {
  fileInput.click();
}

dropzone.addEventListener("click", onDropzoneClick);

dropzone.addEventListener("dragenter", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
});

dropzone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropzone.classList.remove("drag-over");

  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  handleNewFile(file);
});

chooseBtn.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  handleNewFile(file);
});

resetBtn.addEventListener("click", () => {
  currentFile = null;
  uploadInfo.textContent = "";
  clearPreview();
  clearResults();
  fileInput.value = "";
  setState(STATE.IDLE);
  callCleanupIfNeeded();
});

toggleHeatmapBtn.addEventListener("click", () => {
  if (!heatmapImage.src) return;
  currentHeatmapVisible = !currentHeatmapVisible;
  if (currentHeatmapVisible) {
    heatmapImage.classList.add("visible");
    toggleHeatmapBtn.textContent = "Hide decision heatmap";
  } else {
    heatmapImage.classList.remove("visible");
    toggleHeatmapBtn.textContent = "Show decision heatmap";
  }
});

analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) {
    showToast("warning", "No image", "Please select an image first.");
    return;
  }

  setState(STATE.PROCESSING);
  progressBar.classList.remove("hidden");
  clearResults();

  const formData = new FormData();
  formData.append("image", currentFile);

  let timeoutId;
  const controller = new AbortController();
  timeoutId = setTimeout(() => {
    controller.abort();
  }, 45000);

  try {
    const response = await fetch(`${API_BASE_URL}/infer`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errText = await response.text();
      showToast("error", "Analysis failed", errText || "Unknown error.");
      setState(STATE.READY);
      progressBar.classList.add("hidden");
      return;
    }

    const data = await response.json();
    currentInferenceId = data.inference_id || null;

    updateResultsUI(data);
    setState(STATE.DONE);
    progressBar.classList.add("hidden");

    showToast(
      "success",
      "Analysis completed",
      `Predicted class: ${data.pred.label} (confidence ${(data.pred.prob * 100).toFixed(1)}%)`
    );
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      showToast("error", "Timeout", "The request timed out. Please try again.");
    } else {
      showToast("error", "Network error", "Failed to contact the server.");
    }
    setState(STATE.READY);
    progressBar.classList.add("hidden");
  }
});

function updateResultsUI(data) {
  if (!data || !data.pred) return;

  const { label, prob, topk, metrics, uncertainty } = data.pred;

  predLabelEl.textContent = label;
  predProbEl.textContent = `${(prob * 100).toFixed(1)}%`;

  const u = typeof uncertainty === "number" ? uncertainty : 0.0;
  let tagText = "Unknown";
  let cls = "uncertainty-tag";
  if (u <= 0.2) {
    tagText = "High reliability";
    cls += " uncertainty-low";
  } else if (u <= 0.4) {
    tagText = "Medium reliability";
    cls += " uncertainty-medium";
  } else {
    tagText = "Low reliability";
    cls += " uncertainty-high";
  }
  uncertaintyTag.textContent = `${tagText} (uncertainty ${(u * 100).toFixed(
    1
  )}%)`;
  uncertaintyTag.className = cls;

  topkList.innerHTML = "";
  (topk || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "topk-item";

    const labelSpan = document.createElement("span");
    labelSpan.className = "topk-label";
    labelSpan.textContent = item.label;

    const barWrapper = document.createElement("div");
    barWrapper.className = "topk-bar-wrapper";

    const bar = document.createElement("div");
    bar.className = "topk-bar";
    const pct = Math.max(1, Math.min(100, item.prob * 100));
    bar.style.width = `${pct}%`;

    barWrapper.appendChild(bar);

    const valueSpan = document.createElement("span");
    valueSpan.className = "topk-value";
    valueSpan.textContent = `${pct.toFixed(1)}%`;

    row.appendChild(labelSpan);
    row.appendChild(barWrapper);
    row.appendChild(valueSpan);

    topkList.appendChild(row);
  });

  if (metrics) {
    metricRedness.textContent = metrics.redness.toFixed(3);
    metricCyanosis.textContent = metrics.cyanosis.toFixed(3);
    metricArea.textContent = metrics.area_cm2.toFixed(3);
  }

  if (u > 0.4) {
    qualityHint.textContent =
      "The model is uncertain. Try a sharper image with stronger lighting and a consistent distance from the skin.";
  } else {
    qualityHint.textContent =
      "Result looks reasonably reliable, but this demo is not a medical device. Always consult a healthcare professional.";
  }

  const heatUri = data.explain && data.explain.heatmap_uri;
  if (heatUri) {
    heatmapImage.src = `${API_BASE_URL}${heatUri}`;
    heatmapImage.classList.remove("hidden");
    toggleHeatmapBtn.classList.remove("hidden");
    heatmapImage.classList.remove("visible");
    currentHeatmapVisible = false;
    toggleHeatmapBtn.textContent = "Show decision heatmap";
  } else {
    heatmapImage.src = "";
    heatmapImage.classList.add("hidden");
    toggleHeatmapBtn.classList.add("hidden");
  }

  resultsSection.classList.remove("hidden");
}

clearPreview();
clearResults();
setState(STATE.IDLE);
