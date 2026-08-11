(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneIdle = document.getElementById("dropzoneIdle");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImg = document.getElementById("previewImg");
  const analyzeBtn = document.getElementById("analyzeBtn");

  const resultEmpty = document.getElementById("resultEmpty");
  const resultLoading = document.getElementById("resultLoading");
  const resultDone = document.getElementById("resultDone");
  const resultError = document.getElementById("resultError");
  const errorText = document.getElementById("errorText");

  const confidenceArc = document.getElementById("confidenceArc");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceLabel = document.getElementById("confidenceLabel");
  const resultHeadline = document.getElementById("resultHeadline");
  const resultDetail = document.getElementById("resultDetail");

  const resetBtn = document.getElementById("resetBtn");
  const errorResetBtn = document.getElementById("errorResetBtn");
  const langButtons = document.querySelectorAll(".lang-pill-option");

  const RING_CIRCUMFERENCE = 2 * Math.PI * 52; // r=52, matches SVG

  let currentFile = null;
  let currentLang = "en";

  function showResultState(state) {
    [resultEmpty, resultLoading, resultDone, resultError].forEach(el => {
      el.hidden = el !== state;
    });
  }

  function setPreview(file) {
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    dropzoneIdle.hidden = true;
    dropzonePreview.hidden = false;
    currentFile = file;
    analyzeBtn.disabled = false;
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = "";
    dropzoneIdle.hidden = false;
    dropzonePreview.hidden = true;
    previewImg.src = "";
    analyzeBtn.disabled = true;
    showResultState(resultEmpty);
  }

  // --- Upload interactions ---
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      setPreview(fileInput.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setPreview(file);
    }
  });

  // --- Language pill ---
  langButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      langButtons.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      currentLang = btn.dataset.lang;
    });
  });

  // --- Analyze ---
  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    showResultState(resultLoading);
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("language", currentLang);

    try {
      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Something went wrong");
      }

      renderResult(data);
    } catch (err) {
      errorText.textContent = err.message || "Could not reach the analysis service.";
      showResultState(resultError);
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  function renderResult(data) {
    const isAnemic = data.label === "anemic";
    const color = isAnemic ? "#A63446" : "#2F6F62";

    confidenceArc.setAttribute("stroke", color);
    resultHeadline.classList.toggle("is-anemic", isAnemic);
    resultHeadline.classList.toggle("is-non-anemic", !isAnemic);
    resultHeadline.textContent = data.headline;
    resultDetail.textContent = data.detail;
    confidenceLabel.textContent = data.confidence_label;

    const pct = Math.round(data.confidence * 100);
    confidenceValue.textContent = pct + "%";

    // Animate the ring fill on the next frame so the CSS transition fires.
    confidenceArc.setAttribute("stroke-dasharray", `0 ${RING_CIRCUMFERENCE}`);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const filled = (pct / 100) * RING_CIRCUMFERENCE;
        confidenceArc.setAttribute(
          "stroke-dasharray",
          `${filled} ${RING_CIRCUMFERENCE}`
        );
      });
    });

    showResultState(resultDone);
  }

  resetBtn.addEventListener("click", resetUpload);
  errorResetBtn.addEventListener("click", () => showResultState(resultEmpty));
})();
