const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  imageDataUrl: null,
  fileName: null,
  level: 1,
  status: null,
  pendingRecapture: null,
  isRetake: false,
};

const degradationLabels = {
  none: "Без нарушение",
  underexposure: "Недоекспониране",
  overexposure: "Преекспониране",
  gamma: "Гама отклонение",
  color_shift: "Цветови баланс",
  blur: "Размазване",
  jpeg: "JPEG компресия",
  combined: "Комбинирано нарушение",
};

const componentLabels = {
  brightness: "Яркост",
  contrast: "Контраст",
  sharpness: "Острота",
  field_coverage: "Покритие",
  illumination_nonuniformity: "Равномерност",
  color_cast: "Цветови баланс",
  clipping: "Запазени тонове",
};

const stressLabels = {
  none: "Чисти изображения",
  underexposure: "Недоекспониране",
  gamma: "Гама отклонение",
  color_shift: "Цветови баланс",
  blur: "Размазване",
  jpeg: "JPEG компресия",
};

function percent(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function setLoading(loading) {
  const button = $("#analyzeButton");
  button.classList.toggle("loading", loading);
  button.disabled = loading || !state.imageDataUrl;
  $(".button-label").textContent = loading ? "Изчисляване на показателите" : "Анализирай изображението";
}

function setImage(dataUrl, name, options = {}) {
  const { preserveRecapture = false } = options;
  if (!preserveRecapture && !state.pendingRecapture) {
    state.isRetake = false;
    $("#retakeComparison").classList.add("hidden");
  }
  state.imageDataUrl = dataUrl;
  state.fileName = name;
  $("#originalImage").src = dataUrl;
  $("#dropzone").classList.add("ready");
  $("#dropTitle").textContent = name;
  $("#dropMeta").textContent = "Изображението е готово за анализ";
  $("#analyzeButton").disabled = false;
  $("#errorState").classList.add("hidden");
}

function handleFile(file) {
  if (!file) return;
  if (state.pendingRecapture) state.isRetake = true;
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    showError("Поддържат се само JPEG и PNG изображения.");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError("Файлът е по-голям от 20 MB.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => setImage(reader.result, file.name, { preserveRecapture: state.isRetake });
  reader.onerror = () => showError("Файлът не може да бъде прочетен.");
  reader.readAsDataURL(file);
}

function showError(message) {
  $("#errorMessage").textContent = message;
  $("#errorState").classList.remove("hidden");
  $("#emptyState").classList.add("hidden");
  $("#results").classList.add("hidden");
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error("Моделът не е достъпен.");
    state.status = await response.json();
    $("#systemStatus").textContent = "Моделът е готов";
    $(".status-dot").classList.add("ready");
    renderModelMetrics(state.status.metrics);
    renderStressTable(state.status.robustness);
  } catch (error) {
    $("#systemStatus").textContent = "Грешка при модела";
    showError(error.message);
  }
}

async function loadDemo() {
  const button = $("#demoButton");
  button.disabled = true;
  button.textContent = "Зареждане…";
  try {
    const response = await fetch("/api/sample", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Демонстрационният файл липсва.");
    if (state.pendingRecapture) state.isRetake = true;
    setImage(payload.image_data_url, "IDRiD · демонстрационен случай", { preserveRecapture: state.isRetake });
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v14H4zM7 15l3-3 2 2 3-4 2 3"/></svg>Зареди демонстрационно IDRiD изображение`;
  }
}

async function analyze() {
  if (!state.imageDataUrl) return;
  setLoading(true);
  $("#errorState").classList.add("hidden");
  try {
    const degradation = $("#degradation").value;
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data_url: state.imageDataUrl,
        degradation,
        level: state.level,
        normalize: $("#normalization").checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Неизвестна грешка при анализа.");
    renderResult(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
}

function renderResult(result) {
  $("#emptyState").classList.add("hidden");
  $("#errorState").classList.add("hidden");
  $("#results").classList.remove("hidden");
  $("#caseName").textContent = state.fileName || "Текущ случай";

  const degradation = degradationLabels[result.input.degradation] || result.input.degradation;
  const normalized = result.input.normalization ? " · нормализирано" : "";
  $("#analysisTag").textContent = result.input.degradation === "none"
    ? `Оригинален вход${normalized}`
    : `${degradation} · ниво ${result.input.level}${normalized}`;

  const qualityScore = result.quality.score;
  $("#qualityScore").textContent = Math.round(qualityScore);
  $("#qualityRing").style.setProperty("--ring", `${qualityScore}%`);
  $("#qualityGate").textContent = {
    pass: "Технически приемливо",
    review: "Гранично качество",
    fail: "Недостатъчно качество",
  }[result.quality.gate];

  const probability = result.probability_referable_dr;
  const confidence = result.confidence;
  const blocked = result.decision.code === "recapture";
  const reviewOnly = result.decision.code === "review";
  const probabilityCard = $("#drProbability").closest(".metric-card");
  probabilityCard.classList.toggle("blocked", blocked);
  $("#probabilityMetricLabel").textContent = blocked
    ? "Вероятност на модела (блокирана)"
    : reviewOnly
      ? "Вероятност на модела (за преглед)"
      : "Вероятност за реферируема DR";
  $("#probabilityNote").textContent = blocked
    ? "Само за анализ; не участва в решение."
    : "Праг за класификация: 50%";
  $("#drProbability").textContent = percent(probability);
  $("#probabilityBar").style.width = percent(probability);
  $("#confidenceValue").textContent = percent(confidence);
  $("#confidenceBar").style.width = percent(confidence);
  $("#confidenceThreshold").textContent = `Консервативен праг за приемане: ${percent(result.thresholds.confidence, 0)}`;

  const decisionCard = $(".decision-card");
  decisionCard.classList.remove("pass", "fail");
  if (result.decision.code === "recapture") decisionCard.classList.add("fail");
  if (result.decision.code.startsWith("provisional")) decisionCard.classList.add("pass");
  $("#decisionLabel").textContent = result.decision.label;
  $("#decisionReason").textContent = result.decision.reason;

  updateRecaptureFlow(result, degradation, normalized);

  $("#processedImage").src = result.preview;
  $("#contrastImage").src = result.contrast_map;
  $("#originalDimensions").textContent = `${result.input.original_size[0]} × ${result.input.original_size[1]}`;
  $("#processedLabel").textContent = `${degradation}${normalized}`;
  renderQuality(result.quality);

  $("#technicalJson").textContent = JSON.stringify({
    input: result.input,
    raw_quality_metrics: result.raw_quality_metrics,
    thresholds: result.thresholds,
    model: result.model,
    disclaimer: result.disclaimer,
  }, null, 2);

  const motionReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.setTimeout(
    () => $("#results").scrollIntoView({ behavior: motionReduced ? "auto" : "smooth", block: "start" }),
    80,
  );
}

function snapshotResult(result, condition) {
  return {
    quality: result.quality.score,
    decision: result.decision.label,
    probability: result.probability_referable_dr,
    condition,
  };
}

function updateRecaptureFlow(result, degradation, normalized) {
  const recapturePanel = $("#recapturePanel");
  const comparison = $("#retakeComparison");
  const condition = result.input.degradation === "none"
    ? `Оригинален вход${normalized}`
    : `${degradation} · ниво ${result.input.level}${normalized}`;

  if (result.decision.code === "recapture") {
    state.pendingRecapture = snapshotResult(result, condition);
    state.isRetake = false;
    recapturePanel.classList.remove("hidden");
    comparison.classList.add("hidden");
    $("#demoRetakeButton").classList.toggle("hidden", result.input.degradation === "none");
    $("#demoRetakeNote").classList.toggle("hidden", result.input.degradation === "none");
    return;
  }

  recapturePanel.classList.add("hidden");
  if (!state.pendingRecapture || !state.isRetake) {
    comparison.classList.add("hidden");
    return;
  }

  const before = state.pendingRecapture;
  $("#beforeQuality").textContent = `${Math.round(before.quality)}/100`;
  $("#beforeDecision").textContent = `${before.condition} · ${before.decision}`;
  $("#afterQuality").textContent = `${Math.round(result.quality.score)}/100`;
  $("#afterDecision").textContent = `${condition} · ${result.decision.label}`;
  $("#comparisonConclusion").textContent = result.decision.code.startsWith("provisional")
    ? "Новият вход преминава техническата проверка и едва тогава се допуска предварителен моделен резултат. Това не превръща PoC в диагностична система."
    : "Повторното изображение още не позволява автоматичен резултат и остава за човешки преглед.";
  comparison.classList.remove("hidden");
  state.pendingRecapture = null;
  state.isRetake = false;
}

function renderQuality(quality) {
  const host = $("#qualityComponents");
  host.replaceChildren();
  Object.entries(quality.components).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = `component-row ${value < 45 ? "fail" : value < 70 ? "warn" : ""}`;
    const label = document.createElement("span");
    label.textContent = componentLabels[key] || key;
    const track = document.createElement("div");
    track.className = "component-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(1, value)}%`;
    track.appendChild(fill);
    const number = document.createElement("strong");
    number.textContent = Math.round(value);
    row.append(label, track, number);
    host.appendChild(row);
  });

  const issues = $("#qualityIssues");
  if (!quality.issues.length) {
    issues.innerHTML = "Не са открити съществени технически отклонения спрямо обучаващия IDRiD профил.";
  } else {
    const labels = quality.issues.map((item) => `${item.label} (${Math.round(item.score)}/100)`).join(", ");
    issues.innerHTML = `<strong>Провери:</strong> ${labels}.`;
  }
}

function renderModelMetrics(metrics) {
  $("#metricAuroc").textContent = Number(metrics.auroc).toFixed(3);
  $("#metricSensitivity").textContent = percent(metrics.sensitivity, 1);
  $("#metricSpecificity").textContent = percent(metrics.specificity, 1);
  $("#metricEce").textContent = Number(metrics.ece).toFixed(3);
  const ci = metrics.bootstrap_95_ci?.auroc;
  const matrix = metrics.confusion_matrix;
  if (ci && matrix) {
    const [[tn, fp], [fn, tp]] = matrix;
    $("#modelEvidenceNote").textContent =
      `AUROC 95% CI ${Number(ci[0]).toFixed(3)}–${Number(ci[1]).toFixed(3)} · ` +
      `TN ${tn}, FP ${fp}, FN ${fn}, TP ${tp}. Ниската специфичност не е скрита.`;
  }
  drawRiskCoverage(metrics.risk_coverage);
}

function setSeverity(level) {
  state.level = Number(level);
  $$(".segmented button").forEach((button) => {
    const selected = Number(button.dataset.level) === state.level;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function beginRetakeUpload() {
  state.isRetake = true;
  $("#fileInput").value = "";
  $("#fileInput").click();
}

async function runCleanRetakeDemo() {
  if (!state.imageDataUrl) return;
  state.isRetake = true;
  $("#degradation").value = "none";
  $("#normalization").checked = false;
  setSeverity(1);
  await analyze();
}

function drawRiskCoverage(points) {
  const svg = $("#riskChart");
  if (!svg || !points) return;
  const ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const margin = { left: 32, right: 10, top: 10, bottom: 25 };
  const width = 420 - margin.left - margin.right;
  const height = 150 - margin.top - margin.bottom;
  const maxRisk = Math.max(.4, ...points.map((p) => p.risk));
  const x = (value) => margin.left + value * width;
  const y = (value) => margin.top + height - (value / maxRisk) * height;

  [0, .5, 1].forEach((tick) => {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", x(tick)); line.setAttribute("x2", x(tick));
    line.setAttribute("y1", margin.top); line.setAttribute("y2", margin.top + height);
    line.setAttribute("stroke", "rgba(167,205,204,.09)");
    svg.appendChild(line);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", x(tick)); text.setAttribute("y", 145);
    text.setAttribute("text-anchor", "middle"); text.setAttribute("fill", "#617779"); text.setAttribute("font-size", "9");
    text.textContent = `${Math.round(tick * 100)}%`;
    svg.appendChild(text);
  });
  [0, maxRisk / 2, maxRisk].forEach((tick) => {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", margin.left); line.setAttribute("x2", 410);
    line.setAttribute("y1", y(tick)); line.setAttribute("y2", y(tick));
    line.setAttribute("stroke", "rgba(167,205,204,.09)");
    svg.appendChild(line);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", 27); text.setAttribute("y", y(tick) + 3);
    text.setAttribute("text-anchor", "end"); text.setAttribute("fill", "#617779"); text.setAttribute("font-size", "9");
    text.textContent = `${Math.round(tick * 100)}%`;
    svg.appendChild(text);
  });

  const path = document.createElementNS(ns, "path");
  const d = points.map((point, index) => `${index ? "L" : "M"}${x(point.coverage).toFixed(1)},${y(point.risk).toFixed(1)}`).join(" ");
  path.setAttribute("d", d);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "#40d6bd");
  path.setAttribute("stroke-width", "2.5");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);

  points.forEach((point) => {
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", x(point.coverage)); circle.setAttribute("cy", y(point.risk));
    circle.setAttribute("r", "2.5"); circle.setAttribute("fill", "#071015"); circle.setAttribute("stroke", "#40d6bd");
    svg.appendChild(circle);
  });
}

function renderStressTable(rows) {
  const body = $("#stressTableBody");
  if (!body) return;
  body.replaceChildren();
  if (!rows || !rows.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="6">Стрес профилът още не е изчислен.</td>`;
    body.appendChild(row);
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("tr");
    if (item.condition === "none") row.classList.add("baseline-row");
    const deltaCi = item.delta_auroc_ci || [0, 0];
    const deltaEvidence = item.condition === "none"
      ? "контрол"
      : `${Number(item.delta_auroc).toFixed(3)} (${Number(deltaCi[0]).toFixed(3)} до ${Number(deltaCi[1]).toFixed(3)})`;
    const flipCi = item.prediction_flip_rate_ci || [0, 0];
    const flipEvidence = item.condition === "none"
      ? "0.0%"
      : `${percent(item.prediction_flip_rate, 1)} (${percent(flipCi[0], 1)}–${percent(flipCi[1], 1)})`;
    const values = [
      stressLabels[item.condition] || item.condition,
      Number(item.auroc).toFixed(3),
      deltaEvidence,
      Number(item.macro_f1).toFixed(3),
      flipEvidence,
      item.quality_intervention_rate == null ? "—" : percent(item.quality_intervention_rate, 1),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      cell.textContent = value;
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function bindEvents() {
  $("#fileInput").addEventListener("change", (event) => handleFile(event.target.files[0]));
  $("#demoButton").addEventListener("click", loadDemo);
  $("#analyzeButton").addEventListener("click", analyze);
  $("#retakeButton").addEventListener("click", beginRetakeUpload);
  $("#demoRetakeButton").addEventListener("click", runCleanRetakeDemo);
  $$(".segmented button").forEach((button) => {
    button.addEventListener("click", () => setSeverity(button.dataset.level));
  });

  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault(); dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault(); dropzone.classList.remove("drag");
  }));
  dropzone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("#fileInput").click();
    }
  });
}

bindEvents();
loadStatus();
