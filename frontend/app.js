/**
 * RetinaSetu - Automated Clinical Screening & Telemedicine Controller
 * Coaxial AI Retinal Analysis for Diabetic Retinopathy
 */

// Global State
let currentStep = 1;
let currentScanResult = null;
let currentLayer = "original";
let sampleImages = [];
let patientCounter = 841;

let currentPatient = {
  patient_id: `RS-PHC-2026-0${patientCounter}`,
  full_name: "Ramesh Sharma",
  age: 58,
  gender: "Male",
  diabetes_type: "Type 2",
  diabetes_duration: "5 - 10 years",
  hba1c: 8.4,
  blood_pressure: "138/86 mmHg",
  phc_centre: "PHC Khed, District Pune",
  screener_name: "ASHA Worker Sunita Patil (Trained Screener)",
  eye_scanned: "OD (Right Eye)"
};
let activeDoctorCase = null;

// Global Step Navigation
window.goToStep = function(stepNum) {
  currentStep = stepNum;

  // Update Stepper Navigation visuals
  for (let i = 1; i <= 6; i++) {
    const navItem = document.getElementById(`step-nav-${i}`);
    const circle = document.getElementById(`circle-step-${i}`);
    const pane = document.getElementById(`pane-step-${i}`);

    if (navItem) {
      navItem.classList.remove("active", "completed");
      if (i < stepNum) {
        navItem.classList.add("completed");
        if (circle && i <= 4) circle.textContent = "✓";
      } else if (i === stepNum) {
        navItem.classList.add("active");
        if (circle && i <= 4) circle.textContent = i;
      } else {
        if (circle && i <= 4) circle.textContent = i;
      }
    }

    if (pane) {
      pane.classList.remove("active");
      if (i === stepNum) pane.classList.add("active");
    }
  }

  // Update Connectors
  const conn12 = document.getElementById("conn-1-2");
  const conn23 = document.getElementById("conn-2-3");
  const conn34 = document.getElementById("conn-3-4");
  if (conn12) conn12.className = stepNum > 1 ? "step-connector completed" : "step-connector";
  if (conn23) conn23.className = stepNum > 2 ? "step-connector completed" : "step-connector";
  if (conn34) conn34.className = stepNum > 3 ? "step-connector completed" : "step-connector";

  // Trigger lazy loading
  if (stepNum === 4) {
    loadDoctorQueue();
  } else if (stepNum === 5) {
    triggerSimulation();
  } else if (stepNum === 6) {
    loadBenchmarks();
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
};

// Next Patient Reset Workflow
window.promptCloseAndNextPatient = function() {
  const pName = currentPatient.full_name || "Patient";
  const pId = currentPatient.patient_id || "";

  const modalName = document.getElementById("modal-patient-name");
  if (modalName) modalName.textContent = pName;
  const modalId = document.getElementById("modal-patient-id");
  if (modalId) modalId.textContent = pId;

  if (currentScanResult && currentScanResult.grading) {
    const { grading } = currentScanResult;
    const modalStatus = document.getElementById("modal-dr-status");
    if (modalStatus) {
      if (grading.predicted_grade > 0) {
        modalStatus.textContent = "DIABETIC RETINOPATHY: DETECTED (POSITIVE)";
        modalStatus.className = "summary-pill detected";
      } else {
        modalStatus.textContent = "DIABETIC RETINOPATHY: NOT DETECTED (NORMAL)";
        modalStatus.className = "summary-pill normal";
      }
    }

    const modalGrade = document.getElementById("modal-dr-grade");
    if (modalGrade) {
      modalGrade.textContent = `${grading.grade_name} (${grading.confidence}% confidence)`;
    }

    const modalRef = document.getElementById("modal-dr-referral");
    if (modalRef) {
      if (grading.is_referable) {
        modalRef.textContent = "REFERRAL TO OPHTHALMOLOGIST REQUIRED (Grade 2+)";
        modalRef.className = "summary-pill referable";
      } else {
        modalRef.textContent = "NON-REFERABLE CASE (Routine Annual Community Follow-up)";
        modalRef.className = "summary-pill non-referable";
      }
    }
  }

  const modal = document.getElementById("next-patient-modal");
  if (modal) modal.classList.add("open");
};

window.confirmResetAndNextPatient = function() {
  const modal = document.getElementById("next-patient-modal");
  if (modal) modal.classList.remove("open");

  // Increment counter and generate next patient ID
  patientCounter++;
  const nextId = `RS-PHC-2026-0${patientCounter}`;

  // Preserve clinical setting so healthcare worker doesn't re-enter PHC/screener
  const savedPhc = (document.getElementById("p-phc") ? document.getElementById("p-phc").value.trim() : "") || "PHC Khed, District Pune";
  const savedScreener = (document.getElementById("p-screener") ? document.getElementById("p-screener").value.trim() : "") || "ASHA Worker Sunita Patil";

  // Reset current patient to fresh record
  currentPatient = {
    patient_id: nextId,
    full_name: "",
    age: "",
    gender: "Male",
    diabetes_type: "Type 2",
    diabetes_duration: "< 5 years",
    hba1c: "",
    blood_pressure: "120/80 mmHg",
    phc_centre: savedPhc,
    screener_name: savedScreener,
    eye_scanned: "OD (Right Eye)"
  };

  // Populate form with blank values for new patient
  if (document.getElementById("p-id")) document.getElementById("p-id").value = nextId;
  if (document.getElementById("p-name")) document.getElementById("p-name").value = "";
  if (document.getElementById("p-age")) document.getElementById("p-age").value = "";
  if (document.getElementById("p-gender")) document.getElementById("p-gender").value = "Male";
  if (document.getElementById("p-dm-type")) document.getElementById("p-dm-type").value = "Type 2";
  if (document.getElementById("p-dm-duration")) document.getElementById("p-dm-duration").value = "< 5 years";
  if (document.getElementById("p-hba1c")) document.getElementById("p-hba1c").value = "";
  if (document.getElementById("p-bp")) document.getElementById("p-bp").value = "120/80 mmHg";
  if (document.getElementById("p-eye")) document.getElementById("p-eye").value = "OD (Right Eye)";

  // Reset Viewport & Scan state
  currentScanResult = null;
  const baseImg = document.getElementById("base-image");
  if (baseImg) baseImg.src = "";
  const overlayImg = document.getElementById("overlay-image");
  if (overlayImg) {
    overlayImg.src = "";
    overlayImg.style.display = "none";
  }
  const spinMsg = document.getElementById("upload-spinner-msg");
  if (spinMsg) spinMsg.style.display = "none";
  const curFile = document.getElementById("current-filename");
  if (curFile) curFile.textContent = "No image loaded";

  // Update Notification Strip
  document.getElementById("strip-p-name").textContent = "New Patient Registration";
  document.getElementById("strip-p-id").textContent = `(${nextId})`;
  document.getElementById("strip-p-meta").textContent = "• Enter demographics below to begin screening";
  const step2Lbl = document.getElementById("step2-patient-label");
  if (step2Lbl) step2Lbl.textContent = `New Patient (${nextId})`;

  // Automatically navigate back to Step 1
  goToStep(1);

  // Auto-focus the Name field
  setTimeout(() => {
    const nameInput = document.getElementById("p-name");
    if (nameInput) {
      nameInput.focus();
      nameInput.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, 250);
};

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Patient Form
  const patientForm = document.getElementById("patient-form");
  const btnSaveAndProceed = document.getElementById("btn-save-and-proceed");
  const stripBtnNextPatient = document.getElementById("strip-btn-next-patient");
  const patientRecentList = document.getElementById("patient-recent-list");

  function getFormPatientData() {
    return {
      patient_id: document.getElementById("p-id").value.trim() || `RS-PHC-2026-0${patientCounter}`,
      full_name: document.getElementById("p-name").value.trim() || "",
      age: parseInt(document.getElementById("p-age").value) || "",
      gender: document.getElementById("p-gender").value,
      diabetes_type: document.getElementById("p-dm-type").value,
      diabetes_duration: document.getElementById("p-dm-duration").value,
      hba1c: parseFloat(document.getElementById("p-hba1c").value) || 7.5,
      blood_pressure: document.getElementById("p-bp").value.trim() || "130/80 mmHg",
      eye_scanned: document.getElementById("p-eye").value,
      phc_centre: document.getElementById("p-phc").value.trim(),
      screener_name: document.getElementById("p-screener").value.trim()
    };
  }

  function setFormPatientData(p) {
    currentPatient = p;
    document.getElementById("p-id").value = p.patient_id || "";
    document.getElementById("p-name").value = p.full_name || "";
    document.getElementById("p-age").value = p.age || 50;
    document.getElementById("p-gender").value = p.gender || "Male";
    document.getElementById("p-dm-type").value = p.diabetes_type || "Type 2";
    document.getElementById("p-dm-duration").value = p.diabetes_duration || "5 - 10 years";
    document.getElementById("p-hba1c").value = p.hba1c || 7.5;
    document.getElementById("p-bp").value = p.blood_pressure || "130/80 mmHg";
    document.getElementById("p-eye").value = p.eye_scanned || "OD (Right Eye)";
    document.getElementById("p-phc").value = p.phc_centre || "PHC Khed, District Pune";
    document.getElementById("p-screener").value = p.screener_name || "ASHA Worker";
    updatePatientHeaderStrips();
  }

  function updatePatientHeaderStrips() {
    document.getElementById("strip-p-name").textContent = currentPatient.full_name || "New Patient Registration";
    document.getElementById("strip-p-id").textContent = `(${currentPatient.patient_id})`;
    document.getElementById("strip-p-meta").textContent =
      `• Age ${currentPatient.age || "--"} (${currentPatient.gender}) • DM ${currentPatient.diabetes_duration} • HbA1c ${currentPatient.hba1c || "--"}% • ${currentPatient.eye_scanned}`;
    document.getElementById("step2-patient-label").textContent = `${currentPatient.full_name || "Patient"} (${currentPatient.patient_id})`;
    const footerName = document.getElementById("footer-p-name");
    if (footerName) footerName.textContent = currentPatient.full_name || "Patient";
  }

  // Handle form submit (e.g. pressing Enter in any input)
  if (patientForm) {
    patientForm.addEventListener("submit", (e) => {
      e.preventDefault();
      btnSaveAndProceed.click();
    });
  }

  // STEP 1: Save and Automatically Proceed to Step 2
  btnSaveAndProceed.addEventListener("click", async () => {
    const pData = getFormPatientData();
    if (!pData.full_name || pData.full_name.trim() === "") {
      alert("Please enter the patient's full name.");
      document.getElementById("p-name").focus();
      return;
    }
    currentPatient = pData;
    updatePatientHeaderStrips();

    const toast = document.getElementById("patient-save-toast");
    if (toast) {
      toast.style.display = "inline-block";
      toast.textContent = "✓ Patient saved to MongoDB! Proceeding to Step 2 (Image Upload)...";
    }

    try {
      await fetch("/api/patient", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pData)
      });
      loadPatientsList();
    } catch (err) {
      console.error("Save patient error:", err);
    }

    // AUTOMATICALLY PROCEED TO STEP 2 (IMAGE UPLOAD)
    setTimeout(() => {
      goToStep(2);
      if (toast) toast.style.display = "none";
    }, 350);
  });

  if (stripBtnNextPatient) {
    stripBtnNextPatient.addEventListener("click", () => promptCloseAndNextPatient());
  }

  async function loadPatientsList() {
    try {
      const res = await fetch("/api/patients");
      const list = await res.json();
      
      // Enforce strict uniqueness by full_name: same name must not be shown again
      const uniqueList = [];
      const seenNames = new Set();
      list.forEach(p => {
        const nameKey = (p.full_name || "").trim().toLowerCase();
        if (nameKey && !seenNames.has(nameKey)) {
          seenNames.add(nameKey);
          uniqueList.push(p);
        }
      });

      document.getElementById("patient-count-badge").textContent = `${uniqueList.length} Records`;
      patientRecentList.innerHTML = "";

      uniqueList.forEach(p => {
        const item = document.createElement("div");
        item.className = "preset-item";
        item.innerHTML = `
          <div>
            <div class="preset-name">${p.full_name} <span style="font-family: var(--font-mono); font-size: 0.72rem; color: #38bdf8;">${p.patient_id}</span></div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">Age ${p.age} • HbA1c: ${p.hba1c}% • ${p.phc_centre}</div>
          </div>
          <span class="preset-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">Select</span>
        `;
        item.addEventListener("click", () => {
          setFormPatientData(p);
          goToStep(2);
        });
        patientRecentList.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load patients:", err);
    }
  }

  // ==================== STEP 2: UNIVERSAL FILE UPLOADER ====================
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const presetList = document.getElementById("preset-list");
  const uploadSpinner = document.getElementById("upload-spinner-msg");

  dropzone.addEventListener("click", (e) => {
    if (e.target.tagName !== "BUTTON") {
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", e => {
    if (e.target.files && e.target.files.length > 0) {
      processUploadedFile(e.target.files[0]);
    }
  });

  // Paste image handler (Ctrl+V)
  document.addEventListener("paste", e => {
    if (currentStep === 2 && e.clipboardData && e.clipboardData.files.length > 0) {
      processUploadedFile(e.clipboardData.files[0]);
    }
  });

  function processUploadedFile(file) {
    const fname = file.name || "pasted_image.png";
    const isPdf = fname.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
    const label = isPdf ? "PDF Document" : "Image";

    uploadSpinner.style.display = "block";
    uploadSpinner.innerHTML = `<span class="pulse-dot" style="display: inline-block; margin-right: 0.5rem;"></span> Reading and converting ${label} (${fname})...`;

    const reader = new FileReader();
    reader.onload = async function(evt) {
      const base64Data = evt.target.result;
      uploadSpinner.innerHTML = `<span class="pulse-dot" style="display: inline-block; margin-right: 0.5rem;"></span> Running AI Quality Gate, Lesion Segmentation & DR Grading...`;

      try {
        const payload = {
          image_base64: base64Data,
          filename: fname,
          patient_id: currentPatient.patient_id,
          eye_scanned: currentPatient.eye_scanned
        };

        const res = await fetch("/api/screen-upload-b64", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || "Inference error");
        }

        const data = await res.json();
        uploadSpinner.style.display = "none";
        renderScanResult(data, fname);

        // AUTOMATICALLY PROCEED TO STEP 3: AI DIAGNOSIS & GRADING!
        goToStep(3);
      } catch (err) {
        console.error("Upload error:", err);
        uploadSpinner.style.display = "none";
        alert("Upload & AI Analysis Error: " + err.message);
      }
    };
    reader.readAsDataURL(file);
  }

  // Load Presets
  async function loadSamplePresets() {
    try {
      const res = await fetch("/api/sample-images");
      sampleImages = await res.json();
      presetList.innerHTML = "";

      const allPresets = [
        ...sampleImages,
        {
          id: "sample_patient_fundus.pdf",
          name: "Patient Report (PDF Document)",
          grade: 2,
          badge: "PDF Document",
          badge_color: "#a855f7",
          description: "Scanned hospital clinical report with embedded fundus image."
        }
      ];

      allPresets.forEach(s => {
        const item = document.createElement("div");
        item.className = "preset-item";
        item.dataset.id = s.id;
        item.innerHTML = `
          <div>
            <div class="preset-name">${s.name}</div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">${s.description.substring(0, 48)}...</div>
          </div>
          <span class="preset-badge" style="background: ${s.badge_color}22; color: ${s.badge_color}; border: 1px solid ${s.badge_color}55;">
            ${s.badge}
          </span>
        `;
        item.addEventListener("click", () => {
          document.querySelectorAll(".preset-item").forEach(el => el.classList.remove("active"));
          item.classList.add("active");
          triggerPresetScan(s.id);
        });
        presetList.appendChild(item);
      });
    } catch (err) {
      console.error("Preset load error:", err);
    }
  }

  async function triggerPresetScan(presetName) {
    uploadSpinner.style.display = "block";
    uploadSpinner.innerHTML = `<span class="pulse-dot" style="display: inline-block; margin-right: 0.5rem;"></span> Running AI Pipeline on ${presetName}...`;

    try {
      let data;
      if (presetName.endsWith(".pdf")) {
        // Fetch preset PDF as base64
        const resPdf = await fetch(`/sample_data/${presetName}`);
        // If not directly accessible from root, route through backend
        const res = await fetch("/api/screen-preset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preset_name: presetName,
            patient_id: currentPatient.patient_id,
            eye_scanned: currentPatient.eye_scanned
          })
        });
        data = await res.json();
      } else {
        const res = await fetch("/api/screen-preset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preset_name: presetName,
            patient_id: currentPatient.patient_id,
            eye_scanned: currentPatient.eye_scanned
          })
        });
        data = await res.json();
      }

      uploadSpinner.style.display = "none";
      renderScanResult(data, presetName);

      // AUTOMATICALLY PROCEED TO STEP 3!
      goToStep(3);
    } catch (err) {
      console.error("Preset scan error:", err);
      uploadSpinner.style.display = "none";
      alert("Error scanning preset: " + err.message);
    }
  }

  // ==================== STEP 3: RENDER DIAGNOSIS & GRADING ====================
  function renderScanResult(data, filename) {
    currentScanResult = data;
    document.getElementById("current-filename").textContent = `File: ${filename}`;

    const { quality, segmentation, grading, explainability, visualizations } = data;

    // 1. Prominent Binary DR Status + ICDR Grade + Referral Flag
    const verdictContainer = document.getElementById("dr-verdict-container");
    const binaryBadge = document.getElementById("dr-binary-badge");
    const gradeTitle = document.getElementById("dr-grade-title");
    const gradeDesc = document.getElementById("dr-grade-desc");
    const referralBanner = document.getElementById("dr-referral-banner");
    const referralText = document.getElementById("dr-referral-text");

    gradeTitle.textContent = `${grading.grade_name}`;
    gradeDesc.textContent = `${grading.clinical_description} [AI Calibrated Confidence: ${grading.confidence}%]`;

    if (grading.predicted_grade === -1) {
      binaryBadge.className = "dr-status-badge reject";
      binaryBadge.textContent = "QUALITY REJECTION: RECAPTURE REQUIRED";
      verdictContainer.className = "dr-verdict-card reject";
      referralBanner.className = "referral-flag-banner reject";
      referralText.textContent = "QUALITY GATE ALERT: Severe blur or coaxial flash glare. Retake retinal photo.";
    } else if (grading.predicted_grade > 0) {
      binaryBadge.className = "dr-status-badge positive";
      binaryBadge.textContent = "DIABETIC RETINOPATHY: DETECTED (POSITIVE)";
      if (grading.is_referable) {
        verdictContainer.className = "dr-verdict-card referable";
        referralBanner.className = "referral-flag-banner referable";
        referralText.textContent = "REFERRAL TO OPHTHALMOLOGIST REQUIRED (Referable DR - Grade 2+)";
      } else {
        verdictContainer.className = "dr-verdict-card non-referable";
        referralBanner.className = "referral-flag-banner non-referable";
        referralText.textContent = "NON-REFERABLE CASE (Routine Annual Community Follow-Up at PHC)";
      }
    } else {
      binaryBadge.className = "dr-status-badge negative";
      binaryBadge.textContent = "DIABETIC RETINOPATHY: NOT DETECTED (NEGATIVE)";
      verdictContainer.className = "dr-verdict-card non-referable";
      referralBanner.className = "referral-flag-banner non-referable";
      referralText.textContent = "NON-REFERABLE CASE (Routine Annual Community Follow-Up at PHC)";
    }

    // 2. Quality Gate UI
    const qBadge = document.getElementById("quality-verdict-badge");
    qBadge.textContent = quality.verdict;
    qBadge.className = "pill-badge";
    if (quality.verdict === "ACCEPT") {
      qBadge.style.color = "#10b981";
      qBadge.style.borderColor = "#10b981";
    } else if (quality.verdict === "ENHANCED") {
      qBadge.style.color = "#f59e0b";
      qBadge.style.borderColor = "#f59e0b";
    } else {
      qBadge.style.color = "#ef4444";
      qBadge.style.borderColor = "#ef4444";
    }

    const qBox = document.getElementById("quality-feedback-box");
    qBox.textContent = quality.feedback;
    qBox.className = `quality-gate-box ${quality.verdict.toLowerCase()}`;

    document.getElementById("q-score").textContent = quality.quality_score;
    document.getElementById("q-focus").textContent = quality.focus_score;
    document.getElementById("q-glare").textContent = quality.glare_ratio_pct + "%";
    document.getElementById("q-fov").textContent = quality.fov_completeness_pct + "%";

    // 3. Lesion Biomarkers
    document.getElementById("metric-ma").textContent = segmentation.microaneurysm_count;
    document.getElementById("metric-he").textContent = segmentation.exudate_area_pct + "%";
    document.getElementById("metric-hem").textContent = segmentation.hemorrhage_area_pct + "%";
    document.getElementById("metric-nv").textContent = segmentation.neovascularization_index + (segmentation.neovascularization_detected ? " (High Risk)" : " (Normal)");

    // 4. Probability Bars
    const probList = document.getElementById("prob-list");
    probList.innerHTML = "";
    const gradeLabels = [
      "Grade 0 (No DR)",
      "Grade 1 (Mild NPDR)",
      "Grade 2 (Moderate NPDR)",
      "Grade 3 (Severe NPDR)",
      "Grade 4 (Proliferative DR)"
    ];
    grading.class_probabilities.forEach((p, i) => {
      const pct = Math.round(p * 100);
      const isPred = i === grading.predicted_grade;
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-meta">
          <span style="${isPred ? "font-weight: 700; color: #38bdf8;" : "color: var(--text-secondary);"}">${gradeLabels[i]}</span>
          <span style="font-family: var(--font-mono); font-weight: 600;">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pct}%; background: ${isPred ? "linear-gradient(90deg, #06b6d4, #38bdf8)" : "rgba(255,255,255,0.25)"}"></div>
        </div>
      `;
      probList.appendChild(row);
    });

    // 5. Update Viewport
    updateLayerDisplay();

    // 6. Update Doctor Queue details
    syncWithDoctorReview(data);

    // 7. Update Referral Slip Modal
    populateModalData();

    // 8. Update Top Banner Patient Name
    const bannerPName = document.getElementById("banner-patient-name");
    if (bannerPName) bannerPName.textContent = currentPatient.full_name || "Patient";
  }

  // ==================== PACS DIAGNOSTIC TOOLBAR & CONTROLS ====================
  let zoomLevels = [100, 125, 150, 200];
  let currentZoomIdx = 0;
  let isRedFree = false;
  let isInvert = false;
  let isCrosshair = false;

  const viewportEl = document.getElementById("viewport");
  const zoomLabel = document.getElementById("viewport-zoom-label");
  const filterLabel = document.getElementById("viewport-filter-label");
  const crosshairOverlay = document.getElementById("crosshair-overlay");

  function applyViewportState() {
    if (!viewportEl) return;
    // Zoom
    viewportEl.classList.remove("zoom-125", "zoom-150", "zoom-200");
    const lvl = zoomLevels[currentZoomIdx];
    if (lvl === 125) viewportEl.classList.add("zoom-125");
    else if (lvl === 150) viewportEl.classList.add("zoom-150");
    else if (lvl === 200) viewportEl.classList.add("zoom-200");
    if (zoomLabel) zoomLabel.textContent = `Zoom: ${lvl}%`;

    // Red-free (Green Channel Isolation)
    if (isRedFree) {
      viewportEl.classList.add("filter-redfree");
      document.getElementById("tool-red-free")?.classList.add("active");
    } else {
      viewportEl.classList.remove("filter-redfree");
      document.getElementById("tool-red-free")?.classList.remove("active");
    }

    // Invert Contrast
    if (isInvert) {
      viewportEl.classList.add("filter-invert");
      document.getElementById("tool-invert")?.classList.add("active");
    } else {
      viewportEl.classList.remove("filter-invert");
      document.getElementById("tool-invert")?.classList.remove("active");
    }

    // Filter label
    if (filterLabel) {
      if (isRedFree && isInvert) filterLabel.textContent = "Filter: Red-Free + Invert";
      else if (isRedFree) filterLabel.textContent = "Filter: Red-Free (Green)";
      else if (isInvert) filterLabel.textContent = "Filter: Invert Monochrome";
      else filterLabel.textContent = "Filter: Normal";
    }

    // Crosshair reticle
    if (crosshairOverlay) {
      if (isCrosshair) {
        crosshairOverlay.classList.add("active");
        document.getElementById("tool-crosshair")?.classList.add("active");
      } else {
        crosshairOverlay.classList.remove("active");
        document.getElementById("tool-crosshair")?.classList.remove("active");
      }
    }
  }

  document.getElementById("tool-zoom-in")?.addEventListener("click", () => {
    if (currentZoomIdx < zoomLevels.length - 1) {
      currentZoomIdx++;
      applyViewportState();
    }
  });

  document.getElementById("tool-zoom-out")?.addEventListener("click", () => {
    if (currentZoomIdx > 0) {
      currentZoomIdx--;
      applyViewportState();
    }
  });

  document.getElementById("tool-reset-fit")?.addEventListener("click", () => {
    currentZoomIdx = 0;
    isRedFree = false;
    isInvert = false;
    applyViewportState();
  });

  document.getElementById("tool-red-free")?.addEventListener("click", () => {
    isRedFree = !isRedFree;
    applyViewportState();
  });

  document.getElementById("tool-invert")?.addEventListener("click", () => {
    isInvert = !isInvert;
    applyViewportState();
  });

  document.getElementById("tool-crosshair")?.addEventListener("click", () => {
    isCrosshair = !isCrosshair;
    applyViewportState();
  });

  document.getElementById("tool-fullscreen")?.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      viewportEl?.requestFullscreen().catch(err => console.log(err));
    } else {
      document.exitFullscreen().catch(err => console.log(err));
    }
  });

  // Viewport layer switcher
  const layerButtons = document.querySelectorAll(".layer-btn");
  layerButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      layerButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentLayer = btn.dataset.layer;
      updateLayerDisplay();
    });
  });

  function updateLayerDisplay() {
    if (!currentScanResult) return;
    const baseImg = document.getElementById("base-image");
    const overlayImg = document.getElementById("overlay-image");
    const camSlider = document.getElementById("cam-slider-wrap");
    const { visualizations } = currentScanResult;

    if (currentLayer === "original") {
      baseImg.src = visualizations.original;
      overlayImg.style.display = "none";
      if (camSlider) camSlider.style.display = "none";
    } else if (currentLayer === "enhanced") {
      baseImg.src = visualizations.enhanced;
      overlayImg.style.display = "none";
      if (camSlider) camSlider.style.display = "none";
    } else if (currentLayer === "vessels") {
      baseImg.src = visualizations.enhanced;
      overlayImg.src = visualizations.vessel_mask;
      overlayImg.style.display = "block";
      overlayImg.style.opacity = "1";
      if (camSlider) camSlider.style.display = "none";
    } else if (currentLayer === "lesions") {
      baseImg.src = visualizations.enhanced;
      overlayImg.src = visualizations.lesion_mask;
      overlayImg.style.display = "block";
      overlayImg.style.opacity = "1";
      if (camSlider) camSlider.style.display = "none";
    } else if (currentLayer === "gradcam") {
      baseImg.src = visualizations.original;
      overlayImg.src = visualizations.gradcam_overlay;
      overlayImg.style.display = "block";
      const opacityVal = document.getElementById("slider-cam-opacity").value / 100;
      overlayImg.style.opacity = opacityVal;
      if (camSlider) camSlider.style.display = "block";
    }
  }

  const sliderCam = document.getElementById("slider-cam-opacity");
  if (sliderCam) {
    sliderCam.addEventListener("input", e => {
      document.getElementById("val-cam-opacity").textContent = `${e.target.value}%`;
      if (currentLayer === "gradcam") {
        document.getElementById("overlay-image").style.opacity = e.target.value / 100;
      }
    });
  }

  // Buttons in Step 3
  const btnDocSignOff = document.getElementById("btn-open-doctor-signoff");
  if (btnDocSignOff) btnDocSignOff.addEventListener("click", () => goToStep(4));

  const btnNextModal = document.getElementById("btn-next-patient-modal");
  if (btnNextModal) btnNextModal.addEventListener("click", () => promptCloseAndNextPatient());

  const btnViewSlip = document.getElementById("btn-view-slip");
  if (btnViewSlip) {
    btnViewSlip.addEventListener("click", () => {
      populateModalData();
      document.getElementById("report-modal").classList.add("open");
    });
  }

  // Modal handlers
  const btnCloseModal = document.getElementById("btn-close-modal");
  if (btnCloseModal) {
    btnCloseModal.addEventListener("click", () => {
      document.getElementById("report-modal").classList.remove("open");
    });
  }

  function populateModalData() {
    if (!currentScanResult) return;
    const { grading, segmentation } = currentScanResult;
    document.getElementById("rep-patient-name").textContent = currentPatient.full_name;
    document.getElementById("rep-patient-id").textContent = currentPatient.patient_id;
    document.getElementById("rep-patient-demog").textContent = `${currentPatient.age} Yrs / ${currentPatient.gender}`;
    document.getElementById("rep-patient-dm").textContent = `${currentPatient.diabetes_type} (${currentPatient.diabetes_duration}) | HbA1c: ${currentPatient.hba1c}%`;
    document.getElementById("rep-eye-scanned").textContent = currentPatient.eye_scanned;
    document.getElementById("rep-phc").textContent = currentPatient.phc_centre;

    document.getElementById("rep-grade-title").textContent = grading.grade_name;
    document.getElementById("rep-triage-verdict").textContent = grading.triage_category + " — " + grading.referral_action;
    document.getElementById("rep-ma-count").textContent = segmentation.microaneurysm_count;
    document.getElementById("rep-he-pct").textContent = segmentation.exudate_area_pct + "%";
    document.getElementById("rep-hem-pct").textContent = segmentation.hemorrhage_area_pct + "%";
    document.getElementById("rep-confidence").textContent = grading.confidence + "%";

    const repCallout = document.getElementById("rep-callout");
    repCallout.className = `report-callout ${grading.is_referable ? "referable" : "non-referable"}`;
  }

  // ==================== STEP 4: DOCTOR REVIEW QUEUE ====================
  const doctorQueueList = document.getElementById("doctor-queue-list");
  const btnSubmitDoctorSignOff = document.getElementById("btn-submit-doctor-signoff");
  const btnPrintSignedSlip = document.getElementById("btn-print-signed-slip");

  function syncWithDoctorReview(data) {
    activeDoctorCase = data;
    document.getElementById("doc-p-name").textContent = currentPatient.full_name;
    document.getElementById("doc-p-id").textContent = currentPatient.patient_id;
    document.getElementById("doc-ai-finding").textContent = data.grading.grade_name;
    document.getElementById("doc-ai-conf").textContent = data.grading.confidence + "%";
    document.getElementById("doc-biomarkers").textContent =
      `${data.segmentation.microaneurysm_count} MAs, ${data.segmentation.exudate_area_pct}% Hard Exudates, ${data.segmentation.hemorrhage_area_pct}% Hemorrhages`;
    document.getElementById("doc-grade").value = data.grading.predicted_grade;

    const statusBadge = document.getElementById("doc-review-status-badge");
    statusBadge.textContent = data.grading.is_referable ? "Requires Specialist Sign-Off" : "Auto-Cleared (Routine Follow-up)";
    statusBadge.style.color = data.grading.is_referable ? "#f43f5e" : "#10b981";
  }

  async function loadDoctorQueue() {
    try {
      const res = await fetch("/api/doctor/queue");
      const queue = await res.json();
      doctorQueueList.innerHTML = "";

      if (queue.length === 0) {
        doctorQueueList.innerHTML = `<div style="padding: 1rem; color: var(--text-muted); font-size: 0.8rem;">No cases in queue. Complete an AI screening to populate.</div>`;
        return;
      }

      queue.forEach(item => {
        const isPending = item.review_status === "PENDING_DOCTOR_REVIEW";
        const isReviewed = item.review_status === "REVIEWED";
        const el = document.createElement("div");
        el.className = "preset-item";
        el.innerHTML = `
          <div>
            <div class="preset-name">${item.patient_name} <span style="font-size: 0.7rem; color: var(--text-muted);">(${item.patient_id})</span></div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">${item.grading?.grade_name || "Grade --"} • ${item.eye_scanned || "OD"}</div>
          </div>
          <span class="preset-badge" style="background: ${isPending ? "rgba(239, 68, 68, 0.15)" : (isReviewed ? "rgba(16, 185, 129, 0.15)" : "rgba(59, 130, 246, 0.15)")}; color: ${isPending ? "#ef4444" : (isReviewed ? "#10b981" : "#3b82f6")};">
            ${isPending ? "Pending Review" : (isReviewed ? "Signed-Off" : "Auto-Cleared")}
          </span>
        `;
        el.addEventListener("click", () => {
          activeDoctorCase = item;
          document.getElementById("doc-p-name").textContent = item.patient_name;
          document.getElementById("doc-p-id").textContent = item.patient_id;
          document.getElementById("doc-ai-finding").textContent = item.grading?.grade_name || "--";
          document.getElementById("doc-ai-conf").textContent = (item.grading?.confidence || "--") + "%";
          document.getElementById("doc-biomarkers").textContent =
            `${item.segmentation?.microaneurysm_count || 0} MAs, ${item.segmentation?.exudate_area_pct || 0}% Exudates, ${item.segmentation?.hemorrhage_area_pct || 0}% Hemorrhages`;
          if (item.grading) {
            document.getElementById("doc-grade").value = item.grading.predicted_grade;
          }
          if (item.doctor_review) {
            document.getElementById("doc-name").value = item.doctor_review.doctor_name;
            document.getElementById("doc-notes").value = item.doctor_review.clinical_notes;
            document.getElementById("doc-action").value = item.doctor_review.referral_action;
          }
        });
        doctorQueueList.appendChild(el);
      });
    } catch (err) {
      console.error("Queue error:", err);
    }
  }

  btnSubmitDoctorSignOff.addEventListener("click", async () => {
    if (!activeDoctorCase || !activeDoctorCase.screening_id) {
      alert("Please select a screening case first.");
      return;
    }

    const payload = {
      screening_id: activeDoctorCase.screening_id,
      doctor_name: document.getElementById("doc-name").value.trim(),
      confirmed_grade: parseInt(document.getElementById("doc-grade").value),
      clinical_notes: document.getElementById("doc-notes").value.trim(),
      referral_action: document.getElementById("doc-action").value
    };

    try {
      const res = await fetch("/api/doctor/sign-off", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      const msg = document.getElementById("doc-signoff-success-msg");
      msg.style.display = "inline";
      setTimeout(() => { msg.style.display = "none"; }, 3500);

      // Update slip modal
      document.getElementById("rep-doc-name").textContent = payload.doctor_name;
      document.getElementById("rep-doc-action").textContent = payload.referral_action;
      document.getElementById("rep-doc-notes").textContent = payload.clinical_notes;

      loadDoctorQueue();
    } catch (err) {
      console.error("Doctor sign-off error:", err);
      alert("Sign-off error: " + err.message);
    }
  });

  btnPrintSignedSlip.addEventListener("click", () => {
    populateModalData();
    document.getElementById("report-modal").classList.add("open");
  });

  // Filter buttons for queue
  document.getElementById("btn-queue-all").addEventListener("click", () => loadDoctorQueue());
  document.getElementById("btn-queue-pending").addEventListener("click", () => filterQueue("PENDING_DOCTOR_REVIEW"));
  document.getElementById("btn-queue-reviewed").addEventListener("click", () => filterQueue("REVIEWED"));

  async function filterQueue(status) {
    try {
      const res = await fetch(`/api/doctor/queue?status=${status}`);
      const queue = await res.json();
      doctorQueueList.innerHTML = "";
      queue.forEach(item => {
        const el = document.createElement("div");
        el.className = "preset-item";
        el.innerHTML = `
          <div>
            <div class="preset-name">${item.patient_name} <span style="font-size: 0.7rem; color: var(--text-muted);">(${item.patient_id})</span></div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">${item.grading?.grade_name || "Grade --"}</div>
          </div>
          <span class="preset-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8;">View</span>
        `;
        doctorQueueList.appendChild(el);
      });
    } catch (e) {}
  }

  // ==================== STEP 5: SIMULINK SIMULATION ====================
  const simAnnualTarget = document.getElementById("sim-annual-target");
  const simNumPhcs = document.getElementById("sim-num-phcs");
  const simBandwidth = document.getElementById("sim-bandwidth");
  const simTeleDoctors = document.getElementById("sim-tele-doctors");
  const simEdgeTriage = document.getElementById("sim-edge-triage");
  const btnRunSimulation = document.getElementById("btn-run-simulation");

  simAnnualTarget.addEventListener("input", e => {
    document.getElementById("val-annual-target").textContent = Number(e.target.value).toLocaleString();
  });
  simNumPhcs.addEventListener("input", e => {
    document.getElementById("val-num-phcs").textContent = e.target.value;
  });
  simBandwidth.addEventListener("input", e => {
    const val = parseFloat(e.target.value);
    const cat = val < 0.5 ? "2G Rural" : (val < 5 ? "3G Rural" : "4G/5G");
    document.getElementById("val-bandwidth").textContent = `${val} Mbps (${cat})`;
  });
  simTeleDoctors.addEventListener("input", e => {
    document.getElementById("val-tele-doctors").textContent = e.target.value;
  });

  btnRunSimulation.addEventListener("click", triggerSimulation);

  async function triggerSimulation() {
    btnRunSimulation.textContent = "Running Simulink Model...";
    try {
      const payload = {
        annual_target: parseInt(simAnnualTarget.value),
        num_phcs: parseInt(simNumPhcs.value),
        bandwidth_mbps: parseFloat(simBandwidth.value),
        edge_triage: simEdgeTriage.checked,
        num_tele_doctors: parseInt(simTeleDoctors.value)
      };

      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      renderSimulationResults(data);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      btnRunSimulation.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Monte Carlo Queue Simulation</span>';
    }
  }

  function renderSimulationResults(data) {
    document.getElementById("sim-avg-turnaround").textContent = `${data.avg_turnaround_min} min`;
    document.getElementById("sim-p95-turnaround").textContent = `95th Percentile: ${data.p95_turnaround_min} min`;

    const docUtilEl = document.getElementById("sim-doc-util");
    docUtilEl.textContent = `${data.doctor_utilization_pct}%`;
    if (data.doctor_utilization_pct > 85) {
      docUtilEl.style.color = "#ef4444";
    } else {
      docUtilEl.style.color = "#10b981";
    }

    document.getElementById("sim-doc-wait").textContent = `Avg Backlog: ${data.doctor_queue_wait_min} min`;
    document.getElementById("sim-rec-docs").textContent = `${data.recommended_doctors} Specialists (Current: ${data.current_doctors})`;
    document.getElementById("sim-capex-savings").textContent = `₹${data.capex_savings_crores} Crores`;

    // Render Histogram
    const histContainer = document.getElementById("sim-histogram");
    histContainer.innerHTML = "";
    const maxPatients = Math.max(...data.latency_distribution.map(b => b.patients), 1);
    data.latency_distribution.forEach(b => {
      const barHeightPct = Math.max(4, (b.patients / maxPatients) * 100);
      const col = document.createElement("div");
      col.className = "hist-bar-col";
      col.innerHTML = `
        <div class="hist-bar" style="height: ${barHeightPct}%;" title="${b.patients} patients in ${b.range}"></div>
        <div class="hist-label">${b.range}</div>
      `;
      histContainer.appendChild(col);
    });
  }

  // ==================== STEP 6: BENCHMARKS & ACCURACY ====================
  async function loadBenchmarks() {
    try {
      const res = await fetch("/api/benchmarks");
      const data = await res.json();

      const tbody = document.getElementById("benchmark-tbody");
      tbody.innerHTML = "";
      data.datasets.forEach(d => {
        const isRetinaSetu = d.id === "retinasetu";
        const tr = document.createElement("tr");
        if (isRetinaSetu) tr.className = "highlight-row";
        tr.innerHTML = `
          <td><strong>${d.name}</strong></td>
          <td>${d.origin}</td>
          <td>${d.images.toLocaleString()}</td>
          <td style="color: #38bdf8; font-weight: 700;">${d.sensitivity}%</td>
          <td style="color: #10b981; font-weight: 700;">${d.specificity}%</td>
          <td>${d.auc_roc}</td>
          <td>${d.qwk}</td>
        `;
        tbody.appendChild(tr);
      });

      const ablBody = document.getElementById("ablation-tbody");
      ablBody.innerHTML = "";
      data.ablation.forEach(a => {
        const isRetinaSetu = a.model.includes("RetinaSetu");
        const tr = document.createElement("tr");
        if (isRetinaSetu) tr.className = "highlight-row";
        tr.innerHTML = `
          <td><strong>${a.model}</strong></td>
          <td style="color: #38bdf8; font-weight: 700;">${a.sensitivity}%</td>
          <td style="color: #10b981; font-weight: 700;">${a.specificity}%</td>
          <td>${a.qwk}</td>
          <td style="font-size: 0.78rem; color: var(--text-secondary);">${a.clinical_failure}</td>
        `;
        ablBody.appendChild(tr);
      });
    } catch (err) {
      console.error("Failed to load benchmarks:", err);
    }
  }

  // Initial Boot
  updatePatientHeaderStrips();
  loadPatientsList();
  loadSamplePresets();
});
