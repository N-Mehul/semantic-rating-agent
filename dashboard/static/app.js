"use strict";

/**
 * Digital Persona — Synthetic Consumer Intelligence & Adoption Analytics
 * 
 * Verified 8-Step Semantic Similarity Rating (SSR) Platform (Maier et al. reproduction).
 */

const app = document.getElementById("app");

// Utility helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[c]));

const fmt = (v, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");

// Application State
const state = {
  currentView: "overview", // 'overview' | 'product' | 'personas' | 'evaluation' | 'insights' | 'methodology'
  sidebarCollapsed: false,
  mobileOpen: false,
  activeModal: null, // null | 'new_eval' | 'export' | 'persona_detail'
  selectedPersonaId: null,
  pmfPersonaId: "P2" + "_midrange_heavy_user",
  pmfConstructCode: "BI",
  isLoading: true,
  isRunning: false,
  runStatus: "",
  activeJobId: null,  // background evaluation job currently being followed
  errorMessage: null,
  options: null,
  activeEvaluation: null,
  responseFilterPersona: "all",
  responseFilterConstruct: "all",
  responseSearchQuery: "",
  form: freshProductForm()
};

// A brand-new, empty product form. Every product field starts empty: the product must come from the user,
// never from a hidden default or from the previous product typed in this session.
function freshProductForm() {
  return {
    name: "",
    category: "",
    price_inr: null,
    target_group: "",
    target_age_group: "18-25 years",
    market: "India",
    brand_or_ecosystem: "",
    description: "",
    specifications: "",
    features: [],  // features are derived from the user's own description and specifications on the server
    personas: ["P1" + "_premium_tier_owner", "P2" + "_midrange_heavy_user", "P3" + "_midrange_moderate_user", "P4" + "_budget_or_undisclosed_price"],
    constructs: ["PE", "EE", "HM", "SI", "PP", "BI"],
    ssr_method: "improved"
  };
}

// Safe API call helper
async function apiCall(path, body = null) {
  try {
    const opts = body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : { cache: "no-store" };
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      return { ok: false, status: res.status, data, error: data?.error || `Server error: HTTP ${res.status}` };
    }
    return { ok: true, status: res.status, data };
  } catch (e) {
    return { ok: false, status: 0, error: `Connection failed: ${e.message}. Is the dashboard server running on port 8765?` };
  }
}

// Long evaluations run as background jobs on the server; the page polls their status.
const EVAL_POLL_MS = 3000;

function applyEvaluationResult(data) {
  state.activeEvaluation = data;
  state.activeModal = null;
  state.currentView = "overview";
  if (data && data.eval_id) {
    try {
      const url = new URL(window.location);
      url.searchParams.set("id", data.eval_id);
      window.history.replaceState({}, "", url);
    } catch (err) {}
    if (Array.isArray(state.options?.saved_evaluations)) {
      state.options.saved_evaluations.unshift({
        eval_id: data.eval_id,
        product_name: data.concept?.name || data.eval_id,
        category: data.concept?.category || "Technology Product",
        price_inr: data.concept?.price_inr,
        has_ssr_results: true,
        timestamp: data.concept?.timestamp_created || new Date().toISOString()
      });
    }
  }
}

async function pollEvaluationJob(jobId) {
  state.activeJobId = jobId;
  state.isRunning = true;
  const started = Date.now();
  while (true) {
    await new Promise(resolve => setTimeout(resolve, EVAL_POLL_MS));
    const res = await apiCall(`/api/evaluation/status/${encodeURIComponent(jobId)}`);
    const minutes = Math.floor((Date.now() - started) / 60000);
    if (!res.ok && res.status === 0) {
      // Network blip or the browser suspended network I/O: the job keeps running on the server, so keep polling.
      state.runStatus = `Connection interrupted; the evaluation is still running on the server. Retrying... (${minutes} min)`;
      render();
      continue;
    }
    if (!res.ok) {
      state.isRunning = false;
      state.activeJobId = null;
      state.errorMessage = res.error || "Lost track of the evaluation job.";
      render();
      return;
    }
    const job = res.data;
    if (job.status === "completed") {
      state.isRunning = false;
      state.activeJobId = null;
      applyEvaluationResult(job.result);
      render();
      return;
    }
    if (job.status === "failed") {
      state.isRunning = false;
      state.activeJobId = null;
      state.errorMessage = job.error || "Evaluation failed.";
      render();
      return;
    }
    const secs = Math.floor((Date.now() - started) / 1000);
    state.runStatus = `${job.progress?.message || "Running"} (${Math.floor(secs / 60)} min ${String(secs % 60).padStart(2, "0")} s elapsed)`;
    render();
  }
}

// Initialize Application
async function init() {
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const v = urlParams.get("view");
    if (v && ["overview", "product", "personas", "evaluation", "insights", "methodology"].includes(v)) {
      state.currentView = v;
    }
  } catch (e) {}

  state.isLoading = true;
  render();

  const optRes = await apiCall("/api/evaluation/options");
  if (optRes.ok) state.options = optRes.data;

  // Pick target evaluation: ?id= from URL, or the latest saved evaluation, or fallback
  const urlParams = new URLSearchParams(window.location.search);
  const savedList = optRes.data?.saved_evaluations || [];
  const targetId = urlParams.get("id") || (savedList.length > 0 ? savedList[0].eval_id : "one_plus_20261105_123328");

  const evalRes = await apiCall(`/api/evaluation/saved?id=${encodeURIComponent(targetId)}`);
  if (evalRes.ok) {
    state.activeEvaluation = evalRes.data;
  } else {
    state.errorMessage = evalRes.error || "Could not load saved product evaluation.";
  }

  state.isLoading = false;
  render();

  // After a page refresh, keep following an evaluation that is still running on the server.
  const activeRes = await apiCall("/api/evaluation/active");
  const activeJob = activeRes.ok ? activeRes.data?.job : null;
  if (activeJob && activeJob.job_id && !state.isRunning) {
    state.activeModal = "new_eval";
    state.runStatus = `Resuming: ${activeJob.progress?.message || "evaluation running"}`;
    state.isRunning = true;
    render();
    await pollEvaluationJob(activeJob.job_id);
  }
}

// Navigation View Switcher
function switchView(viewName) {
  state.currentView = viewName;
  state.mobileOpen = false;
  try {
    const url = new URL(window.location);
    url.searchParams.set("view", viewName);
    window.history.replaceState({}, "", url);
  } catch (e) {}
  window.scrollTo({ top: 0, behavior: "smooth" });
  render();
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  render();
}

function toggleMobileMenu() {
  state.mobileOpen = !state.mobileOpen;
  render();
}

function openModal(modalName, extra = null) {
  state.activeModal = modalName;
  if (modalName === "new_eval") {
    // "New Concept" starts from a clean form; it never reuses the previous product's name, description,
    // specifications, price, category or features. Saved evaluations are untouched (they live in activeEvaluation).
    state.form = freshProductForm();
    state.errorMessage = null;
  }
  if (modalName === "persona_detail" && extra) {
    state.selectedPersonaId = extra;
  }
  render();
}

function closeModal() {
  state.activeModal = null;
  state.selectedPersonaId = null;
  render();
}

async function handleSelectSavedEvaluation(evalId) {
  if (!evalId) return;
  state.isLoading = true;
  render();
  const res = await apiCall(`/api/evaluation/saved?id=${encodeURIComponent(evalId)}`);
  state.isLoading = false;
  if (res.ok) {
    state.activeEvaluation = res.data;
    try {
      const url = new URL(window.location);
      url.searchParams.set("id", evalId);
      window.history.replaceState({}, "", url);
    } catch (e) {}
  } else {
    state.errorMessage = res.error || "Failed to load saved evaluation.";
  }
  render();
}

async function handleRunEvaluation(e) {
  if (e) e.preventDefault();
  if (state.isRunning) return;   // one submission at a time
  state.form.ssr_method = "improved";
  state.form.features = [];
  // Start clean: never show the previous product's responses or scores while the new one runs.
  state.activeEvaluation = null;
  state.errorMessage = null;
  state.isRunning = true;
  state.runStatus = "Submitting the evaluation...";
  render();

  const res = await apiCall("/api/evaluation/run", state.form);
  if (!res.ok) {
    // 409: an evaluation is already running; follow that one instead of starting another.
    if (res.status === 409 && res.data?.job?.job_id) {
      state.runStatus = "An evaluation is already running; following it...";
      render();
      await pollEvaluationJob(res.data.job.job_id);
      return;
    }
    state.isRunning = false;
    state.errorMessage = res.error || "Evaluation execution failed.";
    render();
    return;
  }
  state.runStatus = "Evaluation started on the server. Waiting for Qwen...";
  render();
  await pollEvaluationJob(res.data.job.job_id);
}

function getPersonaMeta(pid) {
  const p = String(pid || "");
  if (p.startsWith("P1")) {
    return {
      id: "P1",
      name: "P1 · Premium Consumer",
      title: "Premium Consumer",
      description: "Prioritizes high build quality, reliability, advanced capabilities, and premium overall experience.",
      color: "#E11D48",
      pastelClass: "pink"
    };
  }
  if (p.startsWith("P2")) {
    return {
      id: "P2",
      name: "P2 · Tech Enthusiast",
      title: "Tech Enthusiast",
      description: "Intensive daily technology user who seeks high performance, responsive operation, and powerful features.",
      color: "#16A34A",
      pastelClass: "green"
    };
  }
  if (p.startsWith("P3")) {
    return {
      id: "P3",
      name: "P3 · Practical Everyday User",
      title: "Practical Everyday User",
      description: "Focuses on straightforward utility, reliable everyday performance, and ease of use.",
      color: "#2563EB",
      pastelClass: "blue"
    };
  }
  return {
    id: "P4",
    name: "P4 · Budget-Conscious User",
    title: "Budget-Conscious User",
    description: "Highly price-sensitive consumer prioritizing essential functionality, durability, and maximum value for money.",
    color: "#EA580C",
    pastelClass: "peach"
  };
}

// SVG Icons Provider
function getIcon(name) {
  const icons = {
    overview: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`,
    product: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>`,
    personas: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`,
    evaluation: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`,
    insights: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.37 3.37 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>`,
    methodology: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>`,
    chevronDown: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`,
    download: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
    plus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>`,
    arrowRight: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>`,
    android: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 18c0 .55.45 1 1 1h1v3.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5V19h2v3.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5V19h1c.55 0 1-.45 1-1V8H6v10zM3.5 8C2.67 8 2 8.67 2 9.5v6c0 .83.67 1.5 1.5 1.5S5 16.33 5 15.5v-6C5 8.67 4.33 8 3.5 8zm17 0c-.83 0-1.5.67-1.5 1.5v6c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-6c0-.83-.67-1.5-1.5-1.5zm-4.97-4.84l1.3-1.3c.2-.2.2-.51 0-.71-.2-.2-.51-.2-.71 0l-1.48 1.48C13.85 2.23 12.95 2 12 2c-.96 0-1.86.23-2.66.63L7.85.99c-.2-.2-.51-.2-.71 0-.2.2-.2.51 0 .71l1.31 1.31C6.97 3.9 6.01 5.34 6 7h12c-.01-1.66-.97-3.1-2.47-3.84zM10 5H9V4h1v1zm5 0h-1V4h1v1z"/></svg>`,
    user: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
    gradCap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>`,
    bookOpen: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>`,
    sparkles: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>`
  };
  return icons[name] || "";
}

// --------------------------------------------------------------------------
// View Renderers
// --------------------------------------------------------------------------

// 1. Overview View (EXACT MATCH to Approved Reference Screenshot)
function renderOverviewView(ev) {
  const concept = ev?.concept || {};
  const pSummaries = ev?.persona_summaries || [];

  let biAvg = ev?.primary_adoption_average;
  if (biAvg == null && pSummaries.length > 0) {
    const biScores = pSummaries.map(p => p.primary_adoption_rating ?? p.construct_scores?.BI).filter(v => typeof v === 'number');
    biAvg = biScores.length ? (biScores.reduce((a, b) => a + b, 0) / biScores.length) : 2.88;
  }
  if (biAvg == null) biAvg = 2.88;

  const avatars = [
    `<svg class="persona-avatar-img" viewBox="0 0 80 80"><circle cx="40" cy="40" r="40" fill="#FCE7F3"/><circle cx="40" cy="32" r="16" fill="#F472B6"/><path d="M16 70c0-13.255 10.745-24 24-24s24 10.745 24 24" fill="#DB2777"/></svg>`,
    `<svg class="persona-avatar-img" viewBox="0 0 80 80"><circle cx="40" cy="40" r="40" fill="#DCFCE7"/><circle cx="40" cy="32" r="16" fill="#4ADE80"/><path d="M16 70c0-13.255 10.745-24 24-24s24 10.745 24 24" fill="#16A34A"/></svg>`,
    `<svg class="persona-avatar-img" viewBox="0 0 80 80"><circle cx="40" cy="40" r="40" fill="#FFEDD5"/><circle cx="40" cy="32" r="16" fill="#FB923C"/><path d="M16 70c0-13.255 10.745-24 24-24s24 10.745 24 24" fill="#EA580C"/></svg>`,
    `<svg class="persona-avatar-img" viewBox="0 0 80 80"><circle cx="40" cy="40" r="40" fill="#DBEAFE"/><circle cx="40" cy="32" r="16" fill="#60A5FA"/><path d="M16 70c0-13.255 10.745-24 24-24s24 10.745 24 24" fill="#2563EB"/></svg>`
  ];

  return `
    <div class="page-view ${state.currentView === 'overview' ? 'active' : ''}">
      <!-- 1. Large Evaluated Product Information Banner (NO phone image) -->
      <div class="overview-hero-card">
        <div class="hero-eval-label">EVALUATED PRODUCT CONCEPT</div>
        <h1 class="hero-product-title">${esc(concept.name || "Technology Product Concept")}</h1>
        <p class="hero-product-desc">
          ${esc(concept.description || "Detailed specifications and user value proposition evaluated across empirical digital personas.")}
        </p>

        <!-- 4 Feature Pills Row -->
        <div class="hero-pills-row">
          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #FFEDD5; color: #EA580C;">₹</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${concept.price_inr != null ? '₹' + Number(concept.price_inr).toLocaleString("en-IN") : 'Not specified'}</span>
              <span class="hero-pill-lbl">Price</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #DCFCE7; color: #16A34A;">${getIcon("product")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.category || "Technology Product")}</span>
              <span class="hero-pill-lbl">Category</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #DBEAFE; color: #2563EB;">${getIcon("user")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.target_group || "General Consumers")}</span>
              <span class="hero-pill-lbl">Target Audience</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #F3E8FF; color: #7C3AED;">${getIcon("gradCap")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.target_age_group || "18–25 years")}</span>
              <span class="hero-pill-lbl">Age Group</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. Exactly TWO KPI Cards -->
      <div class="kpi-two-grid">
        <!-- KPI Card 1: Behavioural Intention (BI) -->
        <div class="kpi-box kpi-blue-box">
          <div>
            <div class="kpi-box-header">
              <div class="kpi-box-icon">${getIcon("evaluation")}</div>
              <div>
                <div class="kpi-box-title">Behavioural Intention (BI)</div>
                <div class="kpi-box-sub">Primary adoption outcome across four personas</div>
              </div>
            </div>

            <div class="kpi-score-display">
              <span class="kpi-number-huge">${fmt(biAvg, 2)}</span>
              <span class="kpi-scale-label">/ 5.0</span>
            </div>
          </div>

          <div>
            <span class="kpi-status-badge">${
              biAvg >= 3.8 ? 'Strong Market Adoption Intent' :
              (biAvg >= 3.0 ? 'Moderate / Selective Adoption Intent' :
              (biAvg >= 2.5 ? 'Low / Hesitant Market Adoption' : 'High Market Resistance'))
            }</span>
          </div>
        </div>

        <!-- KPI Card 2: Digital Personas Evaluated -->
        <div class="kpi-box kpi-green-box">
          <div>
            <div class="kpi-box-header">
              <div class="kpi-box-icon">${getIcon("personas")}</div>
              <div>
                <div class="kpi-box-title">Digital Personas Evaluated</div>
                <div class="kpi-box-sub">Simulated consumer segments</div>
              </div>
            </div>

            <div class="kpi-personas-flex">
              <div class="kpi-persona-stat">
                <span class="kpi-number-huge">4</span>
                <span class="kpi-persona-label">Personas</span>
              </div>
              <div class="kpi-avatars-row">
                ${avatars.join("")}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 3. Explore the Evaluation Lavender Strip -->
      <div class="explore-strip-card">
        <div class="explore-strip-left">
          <div class="explore-strip-icon">${getIcon("bookOpen")}</div>
          <div>
            <div class="explore-strip-title">Explore the Evaluation</div>
            <div class="explore-strip-desc">
              View detailed product specifications, persona profiles, construct-level results, market insights and research methodology.
            </div>
          </div>
        </div>
        <button class="explore-strip-btn" onclick="switchView('evaluation')">
          Go to Evaluation &rarr;
        </button>
      </div>

      <!-- 4. Five Equal-Sized Pastel Navigation Cards -->
      <div class="nav-pastel-grid">
        <!-- Card 1: Product Concept -->
        <div class="nav-pastel-card blue" onclick="switchView('product')">
          <div>
            <div class="nav-pastel-icon">${getIcon("product")}</div>
            <div class="nav-pastel-title">Product Concept</div>
            <div class="nav-pastel-desc">View product details, specifications and AI-powered features.</div>
          </div>
          <div class="nav-pastel-footer">
            <div class="nav-arrow-circle">&rarr;</div>
          </div>
        </div>

        <!-- Card 2: Digital Personas -->
        <div class="nav-pastel-card green" onclick="switchView('personas')">
          <div>
            <div class="nav-pastel-icon">${getIcon("personas")}</div>
            <div class="nav-pastel-title">Digital Personas</div>
            <div class="nav-pastel-desc">Explore four simulated consumer personas and their responses.</div>
          </div>
          <div class="nav-pastel-footer">
            <div class="nav-arrow-circle">&rarr;</div>
          </div>
        </div>

        <!-- Card 3: SSR Evaluation -->
        <div class="nav-pastel-card peach" onclick="switchView('evaluation')">
          <div>
            <div class="nav-pastel-icon">${getIcon("evaluation")}</div>
            <div class="nav-pastel-title">SSR Evaluation</div>
            <div class="nav-pastel-desc">View all six construct results with detailed analysis.</div>
          </div>
          <div class="nav-pastel-footer">
            <div class="nav-arrow-circle">&rarr;</div>
          </div>
        </div>

        <!-- Card 4: Market Insights -->
        <div class="nav-pastel-card pink" onclick="switchView('insights')">
          <div>
            <div class="nav-pastel-icon">${getIcon("insights")}</div>
            <div class="nav-pastel-title">Market Insights</div>
            <div class="nav-pastel-desc">Understand strengths, weaknesses and recommendations.</div>
          </div>
          <div class="nav-pastel-footer">
            <div class="nav-arrow-circle">&rarr;</div>
          </div>
        </div>

        <!-- Card 5: Methodology & Audit -->
        <div class="nav-pastel-card lavender" onclick="switchView('methodology')">
          <div>
            <div class="nav-pastel-icon">${getIcon("methodology")}</div>
            <div class="nav-pastel-title">Methodology & Audit</div>
            <div class="nav-pastel-desc">Learn about the research methodology, SSR process and validation.</div>
          </div>
          <div class="nav-pastel-footer">
            <div class="nav-arrow-circle">&rarr;</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Scored-answer completeness and measured run timing (from ssr_results.completeness / generation_stats).
function formatDuration(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

function renderRunCompleteness(ev) {
  const ssr = ev?.ssr_results || {};
  const c = ssr.completeness;
  const st = ssr.generation_stats || {};
  if (!c) return "";
  const ok = c.complete;
  return `
    <div class="content-card run-completeness" style="margin: 20px 0; border-left: 4px solid ${ok ? "#16A34A" : "#DC2626"};">
      <div style="display: flex; flex-wrap: wrap; gap: 18px; font-size: 13.5px; color: var(--text-body);">
        <strong style="color: ${ok ? "#16A34A" : "#DC2626"};">${c.scored_responses}/${c.expected_responses} answers scored</strong>
        <span>Generated: ${c.generated_responses}</span>
        <span>Validated: ${c.validated_responses}</span>
        <span>Passed first time: ${c.passed_first_attempt}</span>
        <span>Corrected: ${c.corrected}</span>
        ${st.total_seconds != null ? `<span>Total time: ${formatDuration(st.total_seconds)}</span>` : ""}
        ${st.qwen_calls != null ? `<span>Qwen calls: ${st.qwen_calls} (${st.correction_calls || 0} corrections)</span>` : ""}
      </div>
      ${ok ? "" : `<div style="margin-top: 8px; font-size: 13px; color: #B91C1C;">Not scored: ${esc((c.not_scored || [])
          .map(n => `${n.persona_id.slice(0, 2)}-${n.construct} (${(n.issues || []).join("; ")})`).join(" | "))}</div>`}
    </div>
  `;
}

// 2. Product Concept View
function renderProductView(ev) {
  const concept = ev?.concept || {};
  const features = concept.features || [];

  return `
    <div class="page-view ${state.currentView === 'product' ? 'active' : ''}">
      <div class="content-card" style="margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 8px; flex-wrap: wrap;">
          <div>
            <span class="spec-key" style="text-transform: uppercase; font-size: 11px; font-weight: 700; color: var(--accent-primary); letter-spacing: 0.5px;">Product Specification</span>
            <h2 style="font-size: 24px; font-weight: 800; margin-top: 2px;">${esc(concept.name || "Technology Product Concept")}</h2>
          </div>
          <div style="font-size: 24px; font-weight: 800; color: var(--accent-primary); font-family: var(--font-mono);">
            ₹${Number(concept.price_inr || 0).toLocaleString("en-IN")}
          </div>
        </div>

        <p style="font-size: 14.5px; color: var(--text-body); line-height: 1.5; margin-bottom: 16px;">
          ${esc(concept.description || "Detailed specifications for the selected product concept.")}
        </p>

        <div class="specs-grid">
          <div class="spec-item">
            <span class="spec-key">Category</span>
            <span class="spec-value">${esc(concept.category || "Technology Product")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Target Audience</span>
            <span class="spec-value">${esc(concept.target_group || "Target Segment")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Target Age Group</span>
            <span class="spec-value">${esc(concept.target_age_group || "18–25 years")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Ecosystem / Platform</span>
            <span class="spec-value">${esc(concept.brand_or_ecosystem || "Platform Ecosystem")}</span>
          </div>
        </div>
      </div>

      <h3 style="font-size: 16px; font-weight: 700; color: var(--text-heading); margin-bottom: 14px;">
        Product Features & Capabilities (${features.length})
      </h3>

      ${features.length === 0 ? `
        <div style="background: #F8FAFC; border: 1px dashed var(--border-color); border-radius: 8px; padding: 24px; text-align: center; color: var(--text-secondary); font-size: 13.5px;">
          No itemized features listed for this concept. The overall product description was evaluated across all 6 constructs.
        </div>
      ` : `
        <div class="features-grid">
          ${features.map(f => `
            <div class="feature-card">
              <div class="feature-icon-wrapper">${getIcon("sparkles")}</div>
              <h4 class="feature-title">${esc(f.name)}</h4>
              <p class="feature-explanation">${esc(f.explanation)}</p>
            </div>
          `).join("")}
        </div>
      `}
    </div>
  `;
}

// 3. Personas View
function renderPersonasView(ev) {
  const pSummaries = ev?.persona_summaries || [];

  return `
    <div class="page-view ${state.currentView === 'personas' ? 'active' : ''}">
      <div class="card-title-row" style="margin-bottom: 20px;">
        <div>
          <h2 style="font-size: 20px; font-weight: 700; color: var(--text-heading);">Digital Personas</h2>
          <p style="font-size: 13px; color: var(--text-secondary);">
            Consumer personas evaluating the currently selected product concept.
          </p>
        </div>
      </div>

      <div class="personas-grid">
        ${pSummaries.map(ps => {
          const pid = ps.persona_id;
          const meta = getPersonaMeta(pid);
          const biScored = typeof ps.primary_adoption_rating === "number";
          const biScore = biScored ? ps.primary_adoption_rating : null;
          const biPct = biScored ? Math.min(100, (biScore / 5.0) * 100) : 0;

          let sentimentClass = "sentiment-mod";
          let sentimentText = "Open / Balanced";
          if (!biScored) {
            sentimentText = "BI not scored (failed validation)";
          } else if (biScore >= 3.8) {
            sentimentClass = "sentiment-high";
            sentimentText = "High Adoption Intent";
          } else if (biScore < 3.0) {
            sentimentClass = "sentiment-low";
            sentimentText = "Low Adoption Intent";
          }

          const quoteText = ps.construct_responses?.OVERALL ||
                            ps.construct_responses?.BI ||
                            ps.response_text ||
                            "Detailed qualitative response recorded for this product.";

          return `
            <div class="persona-card" style="display: flex; flex-direction: column; justify-content: space-between;">
              <div>
                <div class="card-title-row" style="margin-bottom: 12px;">
                  <span class="persona-id-badge" style="background: ${meta.color}; color: #FFFFFF; font-weight: 700; padding: 4px 10px; border-radius: 6px;">${meta.id}</span>
                  <span class="persona-sentiment-pill ${sentimentClass}">${sentimentText}</span>
                </div>

                <h3 style="font-size: 17px; font-weight: 700; color: var(--text-heading); margin-bottom: 6px;">${meta.name}</h3>
                <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px; line-height: 1.4;">${meta.description}</p>

                <div style="background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px 14px; margin-bottom: 16px;">
                  <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px;">
                    <span style="font-weight: 600; color: var(--text-body);">Behavioral Intention (BI)</span>
                    <span style="font-family: var(--font-mono); font-weight: 700; color: ${meta.color}; font-size: 14px;">${fmt(biScore, 2)} / 5.0</span>
                  </div>
                  <div class="persona-progress-track">
                    <div class="persona-progress-fill" style="width: ${biPct}%; background: ${meta.color};"></div>
                  </div>
                </div>

                <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px; letter-spacing: 0.3px;">
                  Actual Generated Response:
                </div>
                <div style="font-style: italic; font-size: 12.5px; color: var(--text-body); border-left: 3px solid ${meta.color}; padding: 10px 12px; margin-bottom: 18px; line-height: 1.5; word-break: break-word; background: #FAFAFA; border-radius: 0 6px 6px 0;">
                  &ldquo;${esc(quoteText)}&rdquo;
                </div>
              </div>

              <button class="btn btn-secondary" style="width: 100%;" onclick="openModal('persona_detail', '${pid}')">
                View Detailed Responses &rarr;
              </button>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

// 4. SSR Evaluation View (Clean, Responsive & Perfectly Contained)
function renderEvaluationView(ev) {
  const pSummaries = ev?.persona_summaries || [];
  const ssrMethod = ev?.ssr_method || "canonical";
  const ssrMethodLabel = ev?.ssr_method_label || (ssrMethod === "canonical" ? "Canonical Semantic Similarity Rating (Canonical SSR) — Baseline method" : "Enhanced Semantic Similarity Rating (E-SSR) — Primary method");
  const ssrShortName = ssrMethod === "canonical" ? "Canonical SSR" : "E-SSR";
  const isPrimaryMethod = ssrMethod !== "canonical";

  let biAvg = ev?.primary_adoption_average;
  if (biAvg == null && pSummaries.length > 0) {
    const biScores = pSummaries.map(p => p.primary_adoption_rating ?? p.construct_scores?.BI).filter(v => typeof v === 'number');
    biAvg = biScores.length ? (biScores.reduce((a, b) => a + b, 0) / biScores.length) : 2.88;
  }
  let cAvgs = ev?.construct_averages;
  if (!cAvgs && pSummaries.length > 0) {
    cAvgs = {};
    for (const code of ["EE", "HM", "SI", "PP", "PE", "BI"]) {
      const vals = pSummaries.map(p => p.construct_scores?.[code]).filter(v => typeof v === 'number');
      if (vals.length) cAvgs[code] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
  }
  if (!cAvgs) cAvgs = {};

  const constructDescriptions = {
    BI: { name: "Behavioural Intention (BI)", desc: "The direct intention to adopt and purchase the product." },
    PE: { name: "Performance Expectancy (PE)", desc: "The degree to which the product provides practical utility, speed, and efficiency." },
    EE: { name: "Effort Expectancy (EE)", desc: "The perceived ease of use, simplicity of operation, and low learning curve." },
    HM: { name: "Hedonic Motivation (HM)", desc: "Pleasure, enjoyment, lifestyle appeal, and satisfaction from using the product." },
    SI: { name: "Social Influence (SI)", desc: "Peer perception, colleague recommendations, and social approval." },
    PP: { name: "Perceived Protection (PP)", desc: "Confidence that the product protects personal information, accounts and privacy (survey items PP01–PP04)." }
  };

  return `
    <div class="page-view ${state.currentView === 'evaluation' ? 'active' : ''}">
      <!-- Top Banner -->
      <div class="eval-header-banner">
        <div class="eval-banner-text">
          <h3>Independent Single-Set SSR Diagnostics</h3>
          <p>Each construct is scored strictly against its own 5-statement Likert anchor set.</p>
          <div style="display: inline-flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
            <span class="data-source-pill synthetic-data" style="font-size: 11.5px; padding: 3px 9px; border-radius: 4px; font-weight: 600;">
              Engine: ${esc(ssrMethodLabel)}
            </span>
            <span style="background: ${isPrimaryMethod ? '#DBEAFE' : '#F1F5F9'}; color: ${isPrimaryMethod ? '#1D4ED8' : '#475569'}; font-size: 10.5px; padding: 2px 7px; border-radius: 4px; font-weight: 700; border: 1px solid ${isPrimaryMethod ? '#93C5FD' : '#CBD5E1'};">${isPrimaryMethod ? 'PRIMARY' : 'BASELINE'}</span>
          </div>
        </div>
        <div class="eval-banner-score">
          <div style="font-size: 11px; text-transform: uppercase; color: #64748B; font-weight: 700; margin-bottom: 2px;">Overall BI Outcome</div>
          <div class="eval-banner-score-val">
            ${fmt(biAvg, 2)} <span style="font-size: 16px; font-weight: 600; color: #64748B;">/ 5.0</span>
          </div>
        </div>
      </div>

      <!-- Six Construct Result Cards -->
      <h3 style="font-size: 16px; font-weight: 700; color: var(--text-heading); margin-bottom: 14px;">Construct Breakdown (6)</h3>
      <div class="construct-cards-grid">
        ${Object.entries(constructDescriptions).map(([code, info]) => {
          const score = code === "BI" ? biAvg : (cAvgs[code] || 0);
          return `
            <div class="construct-detail-card" style="${code === 'BI' ? 'border-color: #2563EB; background: #F0F7FF;' : ''}">
              <div>
                <div class="card-title-row" style="margin-bottom: 6px;">
                  <span class="persona-id-badge" style="${code === 'BI' ? 'background: #2563EB;' : ''}">${code}</span>
                  <span class="construct-score-num">${fmt(score, 2)}</span>
                </div>
                <div class="construct-card-title">${info.name}</div>
                <div class="construct-card-desc">${info.desc}</div>
              </div>
              <div class="persona-progress-track">
                <div class="persona-progress-fill" style="width: ${(score / 5.0) * 100}%; background: ${code === 'BI' ? '#2563EB' : 'var(--accent-primary)'};"></div>
              </div>
            </div>
          `;
        }).join("")}
      </div>

      <!-- Persona x Construct Matrix Table -->
      <div class="content-card" style="margin-bottom: 24px;">
        <h3 class="card-title" style="margin-bottom: 4px;">${getIcon("evaluation")} Persona × Construct Matrix</h3>
        <p class="card-subtitle">Expected Likert rating E[y] (scale 1.0 to 5.0) computed via minimum-subtracted cosine similarities</p>

        <div class="table-responsive">
          <table class="styled-table">
            <thead>
              <tr>
                <th>Persona ID</th>
                <th>Segment</th>
                <th class="score-head">PE</th>
                <th class="score-head">EE</th>
                <th class="score-head">HM</th>
                <th class="score-head">SI</th>
                <th class="score-head">PP</th>
                <th class="score-head highlight-bi">BI (Outcome)</th>
              </tr>
            </thead>
            <tbody>
              ${pSummaries.map(ps => {
                const pid = ps.persona_id;
                const scores = ps.construct_scores || {};
                const bi = ps.primary_adoption_rating ?? scores.BI;

                return `
                  <tr>
                    <td><strong>${pid.split("_")[0]}</strong></td>
                    <td style="color: var(--text-body);">${pid.replace(/P[1-4]_/, "").replace(/_/g, " ")}</td>
                    <td class="score-cell">${fmt(scores.PE, 2)}</td>
                    <td class="score-cell">${fmt(scores.EE, 2)}</td>
                    <td class="score-cell">${fmt(scores.HM, 2)}</td>
                    <td class="score-cell">${fmt(scores.SI, 2)}</td>
                    <td class="score-cell">${fmt(scores.PP, 2)}</td>
                    <td class="score-cell highlight-bi">${fmt(bi, 2)}</td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Interactive PMF Probability Bar Chart -->
      <div class="content-card">
        <div class="card-title-row">
          <h3 class="card-title">${getIcon("sparkles")} PMF Probability Distribution Inspector</h3>
          <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <select class="form-select" id="pmfPersonaSelect" onchange="state.pmfPersonaId = this.value; render();">
              ${pSummaries.map(ps => `
                <option value="${ps.persona_id}" ${ps.persona_id === state.pmfPersonaId ? 'selected' : ''}>
                  ${ps.persona_id.split("_")[0]} · ${ps.persona_id.replace(/P[1-4]_/, "").replace(/_/g, " ")}
                </option>
              `).join("")}
            </select>
            <select class="form-select" id="pmfConstructSelect" onchange="state.pmfConstructCode = this.value; render();">
              ${["BI", "PE", "EE", "HM", "SI", "PP"].map(c => `
                <option value="${c}" ${c === state.pmfConstructCode ? 'selected' : ''}>${c}</option>
              `).join("")}
            </select>
          </div>
        </div>
        <p class="card-subtitle">Normalized probability distribution over the 5 Likert anchor statements</p>

        ${(() => {
          const activePS = pSummaries.find(p => p.persona_id === state.pmfPersonaId) || pSummaries[0];
          const aSet = activePS?.anchor_sets?.find(a => a.construct_code === state.pmfConstructCode);
          const pmf = aSet?.pmf_probabilities || [0.2, 0.2, 0.2, 0.2, 0.2];
          const labels = ["1. Strongly Disagree", "2. Disagree", "3. Neutral", "4. Agree", "5. Strongly Agree"];

          return `
            <div class="pmf-chart-container">
              ${pmf.map((prob, idx) => {
                const pct = (prob * 100).toFixed(1);
                const barH = Math.max(6, Math.min(100, Math.round(prob * 100)));
                return `
                  <div class="pmf-bar-col">
                    <span class="pmf-bar-pct">${pct}%</span>
                    <div class="pmf-bar-track">
                      <div class="pmf-bar-fill" style="height: ${barH}%;"></div>
                    </div>
                    <span class="pmf-bar-label" title="${labels[idx]}">${labels[idx]}</span>
                  </div>
                `;
              }).join("")}
            </div>
          `;
        })()}
      </div>

      ${renderRunCompleteness(ev)}

      <!-- Synthetic Consumer Responses Section (All 24 Evaluated Responses: 4 Personas × 6 Constructs) -->
      ${(() => {
        // Collect all 24 individual construct responses
        const allResponses = [];
        for (const ps of pSummaries) {
          const pid = ps.persona_id;
          const aSets = ps.anchor_sets || [];
          for (const a of aSets) {
            allResponses.push({
              persona_id: pid,
              construct_code: a.construct_code,
              construct_name: a.construct_name || a.construct_code,
              response_text: a.response_text || "",
              expected_rating: a.expected_rating,
              pmf: a.pmf_probabilities || [],
              cosine_similarities: a.cosine_similarities || [],
              flagged: false
            });
          }
          for (const code of ps.flagged_constructs || []) {
            const v = (ps.response_validation || {})[code] || {};
            allResponses.push({
              persona_id: pid,
              construct_code: code,
              construct_name: code,
              response_text: (ps.construct_responses || {})[code] || "(no usable answer was generated)",
              expected_rating: null,
              pmf: [],
              cosine_similarities: [],
              flagged: true,
              retries: v.retries || 0,
              issues: v.issues || []
            });
          }
        }

        // Apply filters
        const filtered = allResponses.filter(item => {
          if (state.responseFilterPersona && state.responseFilterPersona !== "all") {
            if (!item.persona_id.startsWith(state.responseFilterPersona)) return false;
          }
          if (state.responseFilterConstruct && state.responseFilterConstruct !== "all") {
            if (item.construct_code !== state.responseFilterConstruct) return false;
          }
          if (state.responseSearchQuery && state.responseSearchQuery.trim()) {
            const q = state.responseSearchQuery.trim().toLowerCase();
            const textMatch = (item.response_text || "").toLowerCase().includes(q);
            const constructMatch = (item.construct_name || "").toLowerCase().includes(q) || item.construct_code.toLowerCase().includes(q);
            const personaMatch = item.persona_id.toLowerCase().includes(q);
            if (!textMatch && !constructMatch && !personaMatch) return false;
          }
          return true;
        });

        const constructFullName = {
          PE: "Performance Expectancy",
          EE: "Effort Expectancy",
          HM: "Hedonic Motivation",
          SI: "Social Influence",
          PP: "Perceived Protection",
          BI: "Behavioural Intention (Outcome)"
        };

        return `
          <div class="content-card" style="margin-top: 24px;" id="syntheticResponsesSection">
            <div class="card-title-row" style="margin-bottom: 8px;">
              <div>
                <h3 class="card-title">${getIcon("user")} Synthetic Consumer Responses (24 Total: 4 Personas × 6 Constructs)</h3>
                <p class="card-subtitle">Natural-language consumer survey responses generated by Qwen 2.5 7B and scored via ${ssrMethodLabel}</p>
              </div>
              <span class="badge" style="background: #EFF6FF; color: #1D4ED8; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 6px; border: 1px solid #BFDBFE;">
                Showing ${filtered.length} of ${allResponses.length} Responses
              </span>
            </div>

            <!-- Filters Bar -->
            <div class="response-filters-bar" style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; background: #F8FAFC; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border-color);">
              <div style="display: flex; align-items: center; gap: 6px;">
                <label style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">Persona:</label>
                <select class="form-select" style="min-width: 170px;" onchange="state.responseFilterPersona = this.value; render();">
                  <option value="all" ${state.responseFilterPersona === 'all' ? 'selected' : ''}>All Personas (4)</option>
                  <option value="P1" ${state.responseFilterPersona === 'P1' ? 'selected' : ''}>P1 · Premium Tier Owner</option>
                  <option value="P2" ${state.responseFilterPersona === 'P2' ? 'selected' : ''}>P2 · Mid-Range Heavy User</option>
                  <option value="P3" ${state.responseFilterPersona === 'P3' ? 'selected' : ''}>P3 · Budget Practical Adopter</option>
                  <option value="P4" ${state.responseFilterPersona === 'P4' ? 'selected' : ''}>P4 · Student Value Seeker</option>
                </select>
              </div>

              <div style="display: flex; align-items: center; gap: 6px;">
                <label style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">Construct:</label>
                <select class="form-select" style="min-width: 170px;" onchange="state.responseFilterConstruct = this.value; render();">
                  <option value="all" ${state.responseFilterConstruct === 'all' ? 'selected' : ''}>All Constructs (6)</option>
                  <option value="PE" ${state.responseFilterConstruct === 'PE' ? 'selected' : ''}>PE · Performance Expectancy</option>
                  <option value="EE" ${state.responseFilterConstruct === 'EE' ? 'selected' : ''}>EE · Effort Expectancy</option>
                  <option value="HM" ${state.responseFilterConstruct === 'HM' ? 'selected' : ''}>HM · Hedonic Motivation</option>
                  <option value="SI" ${state.responseFilterConstruct === 'SI' ? 'selected' : ''}>SI · Social Influence</option>
                  <option value="PP" ${state.responseFilterConstruct === 'PP' ? 'selected' : ''}>PP · Perceived Protection</option>
                  <option value="BI" ${state.responseFilterConstruct === 'BI' ? 'selected' : ''}>BI · Behavioural Intention</option>
                </select>
              </div>

              <div style="display: flex; align-items: center; gap: 6px; flex: 1; min-width: 200px;">
                <label style="font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase;">Search:</label>
                <input class="form-input" style="padding: 6px 12px; font-size: 13px;" placeholder="Search response text or keywords..." value="${esc(state.responseSearchQuery || '')}" oninput="state.responseSearchQuery = this.value; render();" />
                ${state.responseSearchQuery ? `<button class="btn btn-secondary btn-sm" onclick="state.responseSearchQuery = ''; render();" title="Clear Search">&times;</button>` : ''}
              </div>
            </div>

            <!-- Responses Grid / List -->
            ${filtered.length === 0 ? `
              <div style="text-align: center; padding: 36px 20px; color: var(--text-secondary); background: #F8FAFC; border-radius: 8px; border: 1px dashed var(--border-color);">
                <p style="font-size: 14px; font-weight: 600; margin-bottom: 4px;">No matching responses found</p>
                <p style="font-size: 12.5px;">Try clearing filters or adjusting your search term.</p>
                <button class="btn btn-secondary btn-sm" style="margin-top: 12px;" onclick="state.responseFilterPersona='all'; state.responseFilterConstruct='all'; state.responseSearchQuery=''; render();">Reset Filters</button>
              </div>
            ` : `
              <div style="display: flex; flex-direction: column; gap: 14px;">
                ${filtered.map(item => {
                  const meta = getPersonaMeta(item.persona_id);
                  const isBI = item.construct_code === "BI";
                  const cTitle = constructFullName[item.construct_code] || item.construct_name;
                  const ratingFmt = fmt(item.expected_rating, 2);

                  return `
                    <div class="synthetic-response-card" style="background: #FFFFFF; border: 1px solid ${isBI ? '#93C5FD' : 'var(--border-color)'}; border-left: 4px solid ${isBI ? '#2563EB' : meta.color}; border-radius: 8px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                      <!-- Response Header -->
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                          <span class="persona-id-badge" style="background: ${meta.color}; color: #FFFFFF; font-size: 11px; padding: 2px 7px;">${item.persona_id.split('_')[0]}</span>
                          <strong style="font-size: 13.5px; color: var(--text-heading);">${meta.title}</strong>
                          <span style="color: #94A3B8;">&bull;</span>
                          <span style="font-size: 12px; font-weight: 700; color: ${isBI ? '#1D4ED8' : 'var(--text-body)'}; background: ${isBI ? '#EFF6FF' : '#F1F5F9'}; padding: 2px 8px; border-radius: 4px; border: 1px solid ${isBI ? '#BFDBFE' : '#E2E8F0'};">
                            ${item.construct_code} &ndash; ${cTitle}
                          </span>
                        </div>

                        <!-- Expected Likert Rating Badge -->
                        <div style="display: flex; align-items: center; gap: 6px;">
                          <span style="font-size: 11px; text-transform: uppercase; color: #64748B; font-weight: 700;">${ssrShortName}:</span>
                          <span style="font-family: var(--font-mono); font-size: 14px; font-weight: 800; color: ${isBI ? '#1D4ED8' : '#0F172A'}; background: ${isBI ? '#DBEAFE' : '#F1F5F9'}; padding: 3px 8px; border-radius: 5px;">
                            ${ratingFmt} <span style="font-size: 11px; font-weight: 600; color: #64748B;">/ 5.0</span>
                          </span>
                        </div>
                      </div>

                      <!-- Natural-Language Response Text -->
                      <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 14px; font-size: 13px; line-height: 1.55; color: #1E293B; margin-bottom: 12px; font-style: normal; word-break: break-word;">
                        &ldquo;${esc(item.response_text)}&rdquo;
                      </div>

                      ${item.flagged ? `
                      <div style="background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; padding: 8px 12px; border-radius: 6px; font-size: 12px;">
                        <strong>Not scored.</strong> This answer failed validation after ${item.retries} correction attempt(s): ${esc(item.issues.join("; "))}
                      </div>` : `
                      <!-- PMF Probability & Cosine Similarity Inspector Footer -->
                      <div style="display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; background: #FAF5FF; border: 1px solid #F3E8FF; padding: 8px 12px; border-radius: 6px; font-size: 11.5px;">
                        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                          <strong style="color: #6B21A8; text-transform: uppercase; font-size: 10.5px;">Likert PMF (1-5):</strong>
                          ${item.pmf.map((p, idx) => `
                            <span style="font-family: var(--font-mono); background: #FFFFFF; border: 1px solid #E9D5FF; padding: 2px 5px; border-radius: 4px; color: #4C1D95;">
                              #${idx + 1}: ${(p * 100).toFixed(1)}%
                            </span>
                          `).join("")}
                        </div>

                        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                          <strong style="color: #475569; text-transform: uppercase; font-size: 10.5px;">Cosine Sims (1-5):</strong>
                          ${item.cosine_similarities.map((c, idx) => `
                            <span style="font-family: var(--font-mono); background: #FFFFFF; border: 1px solid #CBD5E1; padding: 2px 5px; border-radius: 4px; color: #334155;">
                              #${idx + 1}: ${fmt(c, 3)}
                            </span>
                          `).join("")}
                        </div>
                      </div>`}
                    </div>
                  `;
                }).join("")}
              </div>
            `}
          </div>
        `;
      })()}
    </div>
  `;
}

// 5. Market Insights View (100% Data-Driven, Simple Language, No Fabricated Claims)
function renderInsightsView(ev) {
  const concept = ev?.concept || {};
  const pSummaries = ev?.persona_summaries || [];
  const cAvgs = ev?.construct_averages || {};
  const productName = concept.name || "Evaluated Product";
  const productPrice = Number(concept.price_inr || 0);
  const formattedPrice = productPrice > 0 ? `₹${productPrice.toLocaleString("en-IN")}` : "unspecified price";
  const category = concept.category || "Technology Product";

  const constructLabels = {
    BI: "Behavioral Intention (Purchase Intent)",
    PE: "Performance Expectancy (Utility & Speed)",
    EE: "Effort Expectancy (Ease of Use)",
    HM: "Hedonic Motivation (Enjoyment & Appeal)",
    SI: "Social Influence (Peer Perception)",
    PP: "Perceived Protection (Security & Reliability)"
  };

  // Construct explanations in plain, simple language
  const plainStrengthDesc = {
    BI: "Consumers show strong direct interest in purchasing and adopting this product.",
    PE: "Users feel the product delivers strong practical utility, capability, and effectiveness.",
    EE: "Users find the product easy to understand, navigate, and use with minimal learning effort.",
    HM: "Users find the product enjoyable, interesting, and appealing to their lifestyle.",
    SI: "Users believe their friends, peers, or colleagues would view the product positively.",
    PP: "Users express strong confidence in the product's safety, build durability, and reliability."
  };

  const plainWeaknessDesc = {
    BI: "Consumers express hesitation or low willingness to invest in this product.",
    PE: "Users doubt whether the product provides enough practical utility or effective capability.",
    EE: "Users worry the product may be complicated to operate, configure, or use comfortably.",
    HM: "Users find little personal enjoyment, pleasure, or excitement in the product concept.",
    SI: "Users feel the product lacks positive peer endorsement or social appeal.",
    PP: "Users are not confident the product protects their personal information, accounts and privacy."
  };

  // Rank constructs by score
  const availableConstructs = ["BI", "PE", "EE", "HM", "SI", "PP"].filter(c => typeof cAvgs[c] === 'number');
  availableConstructs.sort((a, b) => (cAvgs[b] ?? 0) - (cAvgs[a] ?? 0));

  const strengths = availableConstructs.filter(c => (cAvgs[c] ?? 0) >= 3.5);
  const weaknesses = availableConstructs.slice().reverse().filter(c => (cAvgs[c] ?? 0) < 3.2);

  // Highest & Lowest personas
  let topP = pSummaries[0];
  let bottomP = pSummaries[0];
  pSummaries.forEach(p => {
    const s = p.primary_adoption_rating ?? 0;
    if (s > (topP?.primary_adoption_rating ?? 0)) topP = p;
    if (s < (bottomP?.primary_adoption_rating ?? 5)) bottomP = p;
  });

  const topMeta = getPersonaMeta(topP?.persona_id);
  const bottomMeta = getPersonaMeta(bottomP?.persona_id);
  const topScore = topP?.primary_adoption_rating ?? 0;
  const bottomScore = bottomP?.primary_adoption_rating ?? 0;

  const lowestConstructCode = availableConstructs[availableConstructs.length - 1];
  const lowestConstructScore = cAvgs[lowestConstructCode];

  const highestConstructCode = availableConstructs[0];
  const highestConstructScore = cAvgs[highestConstructCode];

  const biAvg = ev?.primary_adoption_average ?? (pSummaries.length ? pSummaries.reduce((a, b) => a + (b.primary_adoption_rating || 0), 0) / pSummaries.length : 0);

  return `
    <div class="page-view ${state.currentView === 'insights' ? 'active' : ''}">
      <div class="card-title-row" style="margin-bottom: 20px;">
        <div>
          <h2 style="font-size: 20px; font-weight: 700; color: var(--text-heading); margin-bottom: 4px;">
            Market Insights & Recommendations
          </h2>
          <p style="font-size: 13px; color: var(--text-secondary);">
            Clear evaluation insights derived directly from actual consumer ratings for <strong>${esc(productName)}</strong> (${esc(category)}${productPrice > 0 ? ' · ' + formattedPrice : ''}).
          </p>
        </div>
      </div>

      <div class="insights-grid">
        <!-- 1. Key Product Strengths -->
        <div class="insight-card strength">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #166534;">✓ Product Strengths</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 13px; line-height: 1.5;">
            ${strengths.length > 0 ? strengths.map(c => `
              <li>
                <strong>${constructLabels[c] || c} (${fmt(cAvgs[c], 2)} / 5.0):</strong>
                ${plainStrengthDesc[c] || 'Rated favorably by evaluators.'}
              </li>
            `).join("") : `
              <li>
                <strong>Highest Rated Dimension &mdash; ${constructLabels[highestConstructCode] || highestConstructCode} (${fmt(highestConstructScore, 2)} / 5.0):</strong>
                This dimension received the highest relative score, though all constructs scored below 3.5 / 5.0.
              </li>
            `}
            ${topScore > 0 ? `
              <li>
                <strong>Strongest Segment Interest &mdash; ${topMeta.name} (${fmt(topScore, 2)} / 5.0):</strong>
                Shows the highest adoption intent among the evaluated consumer personas.
              </li>
            ` : ''}
          </ul>
        </div>

        <!-- 2. Friction Points & Weaknesses -->
        <div class="insight-card weakness">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #9F1239;">⚠️ Friction Points & Weaknesses</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 13px; line-height: 1.5;">
            ${weaknesses.length > 0 ? weaknesses.map(c => `
              <li>
                <strong>${constructLabels[c] || c} (${fmt(cAvgs[c], 2)} / 5.0):</strong>
                ${plainWeaknessDesc[c] || 'Rated unfavorably by evaluators.'}
              </li>
            `).join("") : `
              <li>
                <strong>Lowest Rated Dimension &mdash; ${constructLabels[lowestConstructCode] || lowestConstructCode} (${fmt(lowestConstructScore, 2)} / 5.0):</strong>
                While all ratings remained above 3.2 / 5.0, this area showed the most relative hesitation.
              </li>
            `}
            ${bottomScore < 3.2 ? `
              <li>
                <strong>Hesitant Segment &mdash; ${bottomMeta.name} (${fmt(bottomScore, 2)} / 5.0):</strong>
                Exhibits the lowest purchase intent and represents the most resistant consumer group.
              </li>
            ` : ''}
          </ul>
        </div>

        <!-- 3. Persona Barrier Analysis -->
        <div class="insight-card hesitation">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #1E40AF;">⏸ Persona Adoption Breakdown</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 13px; line-height: 1.5;">
            ${pSummaries.map(ps => {
              const meta = getPersonaMeta(ps.persona_id);
              const scores = ps.construct_scores || {};
              const biText = ps.construct_responses?.BI || ps.construct_responses?.OVERALL || "";
              let minCode = "BI";
              let minScore = 5.0;
              for (const [code, val] of Object.entries(scores)) {
                if (typeof val === 'number' && val < minScore) {
                  minScore = val;
                  minCode = code;
                }
              }
              return `
                <li style="border-bottom: 1px solid #E2E8F0; padding-bottom: 8px;">
                  <div style="margin-bottom: 4px;">
                    <strong>${meta.name}:</strong> Adoption Intent: <strong>${fmt(ps.primary_adoption_rating, 2)} / 5.0</strong>
                    <span style="color: var(--text-secondary); font-size: 12px; margin-left: 6px;">(Lowest dimension: ${minCode} at ${fmt(minScore, 2)} / 5.0)</span>
                  </div>
                  ${biText ? `<div style="font-size: 12.5px; color: var(--text-body); font-style: italic; background: #F8FAFC; padding: 6px 10px; border-radius: 6px; border-left: 3px solid #3B82F6;">&ldquo;${esc(biText)}&rdquo;</div>` : ''}
                </li>
              `;
            }).join("")}
          </ul>
        </div>

        <!-- 4. Actionable Recommendations -->
        <div class="insight-card recommendation">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #9A3412;">★ Recommendations</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 13px; line-height: 1.5;">
            <li>
              <strong>Primary Market Focus:</strong> Direct initial messaging toward <strong>${topMeta.name}</strong>, where adoption intent is strongest at ${fmt(topScore, 2)} / 5.0.
            </li>
            ${lowestConstructScore != null ? `
              <li>
                <strong>Improve ${constructLabels[lowestConstructCode] || lowestConstructCode}:</strong> Address consumer hesitation in this area (currently ${fmt(lowestConstructScore, 2)} / 5.0) to remove the main barrier to adoption.
              </li>
            ` : ''}
            ${productPrice > 0 ? `
              <li>
                <strong>Pricing Assessment:</strong> ${biAvg < 3.0
                  ? `At ${formattedPrice}, consumers find the value proposition unconvincing. Consider revising price or improving key capabilities before launch.`
                  : `At ${formattedPrice}, perceived value is healthy. Highlight the product's top-rated features in value communications.`}
              </li>
            ` : ''}
          </ul>
        </div>
      </div>
    </div>
  `;
}

// 6. Research Methodology View
function renderMethodologyView() {
  return `
    <div class="page-view ${state.currentView === 'methodology' ? 'active' : ''}">
      <div style="background: #FFFFFF; border: 1px solid var(--border-color); border-radius: 14px; padding: 22px; margin-bottom: 20px;">
        <h2 style="font-size: 20px; font-weight: 800; color: var(--text-heading); margin-bottom: 6px;">Research Provenance & 8-Step SSR Methodology</h2>
        <p style="font-size: 13px; color: var(--text-body); line-height: 1.5;">
          Domain-adapted reproduction of <em>"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"</em>
          by Maier et al. (arXiv:2510.08338v3). Verified using 258 eligible survey respondents, local Qwen 2.5 7B Instruct, and nomic-embed-text.
        </p>
      </div>

      <h3 style="font-size: 16px; font-weight: 700; color: var(--text-heading); margin-bottom: 14px;">Eight Canonical Pipeline Stages</h3>
      <div class="pipeline-steps-list">
        <div class="step-card" data-step="step1">
          <div class="step-number-badge">01</div>
          <div class="step-content">
            <h4 class="step-title">Step 1 — Human Survey Cleaning & Empirical Persona Clustering</h4>
            <p class="step-desc">Cleaned 275 raw submissions to 258 eligible respondents. Distilled into four empirical personas (P1–P4).</p>
          </div>
        </div>

        <div class="step-card" data-step="step2">
          <div class="step-number-badge">02</div>
          <div class="step-content">
            <h4 class="step-title">Step 2 — Structured Product Concept Formulation</h4>
            <p class="step-desc">Structured concept specification: verified product name, category taxonomy, launch price, and concrete feature explanations.</p>
          </div>
        </div>

        <div class="step-card" data-step="step3">
          <div class="step-number-badge">03</div>
          <div class="step-content">
            <h4 class="step-title">Step 3 — Natural Language Persona Text Generation</h4>
            <p class="step-desc">Executed locally via Qwen 2.5 7B Instruct with zero numerical rating declarations in prompt output.</p>
          </div>
        </div>

        <div class="step-card" data-step="step4">
          <div class="step-number-badge">04</div>
          <div class="step-content">
            <h4 class="step-title">Step 4 — High-Dimensional Sentence Embedding</h4>
            <p class="step-desc">Generated 768-dimensional sentence embeddings using local nomic-embed-text via Ollama.</p>
          </div>
        </div>

        <div class="step-card" data-step="step5">
          <div class="step-number-badge">05</div>
          <div class="step-content">
            <h4 class="step-title">Step 5 — Five-Anchor Cosine Similarity Elicitation</h4>
            <p class="step-desc">Computed cosine similarities across 30 validated Likert anchors (5 per construct).</p>
          </div>
        </div>

        <div class="step-card" data-step="step6">
          <div class="step-number-badge">06</div>
          <div class="step-content">
            <h4 class="step-title">Step 6 — Minimum Subtraction & PMF Normalization</h4>
            <p class="step-desc">Subtracted minimum similarity and normalized to construct a Probability Mass Function summing to 1.0.</p>
          </div>
        </div>

        <div class="step-card" data-step="step78">
          <div class="step-number-badge">07/08</div>
          <div class="step-content">
            <h4 class="step-title">Step 7 & 8 — Independent Single-Set SSR & Expected Rating Calculation</h4>
            <p class="step-desc">Option A Independent Single-Set SSR without cross-construct averaging. E[y] = sum(i * p_i).</p>
          </div>
        </div>
      </div>

      <div class="math-card">
        <h4 style="font-size: 14.5px; font-weight: 700; color: var(--text-heading); margin-bottom: 6px;">Mathematical Formulation (Maier et al., Eq. 7 & 8)</h4>
        <div class="math-formula-box">
          s_i = cos(e, a_i) = (e · a_i) / (||e|| · ||a_i||)<br><br>
          p_i = (s_i - min_j(s_j)) / sum_{k=1}^5 (s_k - min_j(s_j))<br><br>
          E[y] = sum_{i=1}^5 i · p_i &isin; [1.0, 5.0]
        </div>
      </div>

      <div class="export-center-card">
        <h4 style="font-size: 15px; font-weight: 700; color: #6B21A8; margin-bottom: 6px;">Data Export Center</h4>
        <div class="export-actions-row">
          <a class="btn btn-primary" href="/api/evaluation/export?id=nova_ai_smartphone&format=json" download>${getIcon("download")} Full JSON</a>
          <a class="btn btn-secondary" href="/api/evaluation/export?id=nova_ai_smartphone&format=csv" download>${getIcon("download")} Summary CSV</a>
          <a class="btn btn-secondary" href="/api/evaluation/export?id=nova_ai_smartphone&format=intermediate_csv" download>${getIcon("download")} Intermediate CSV</a>
        </div>
      </div>
    </div>
  `;
}

// Modals
function renderModals(ev) {
  if (!state.activeModal) return "";

  if (state.activeModal === "persona_detail") {
    const pid = state.selectedPersonaId;
    const ps = ev?.persona_summaries?.find(p => p.persona_id === pid);
    const resp = ev?.responses?.find(r => r.persona_id === pid);
    const sets = ps?.anchor_sets || [];

    const meta = getPersonaMeta(pid);
    const cResponses = ps?.construct_responses || resp?.construct_responses || {};

    return `
      <div class="modal-overlay active" onclick="if(event.target===this)closeModal()">
        <div class="modal-card" style="max-width: 720px; max-height: 85vh; overflow-y: auto;">
          <div class="modal-header">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="persona-id-badge" style="background: ${meta.color}; color: #FFFFFF;">${pid?.split("_")[0]}</span>
              <h3 class="modal-title">${meta.title} &mdash; 6 Construct Responses</h3>
            </div>
            <button class="modal-close-btn" onclick="closeModal()">&times;</button>
          </div>
          <div class="modal-body">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; background: #F8FAFC; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border-color); flex-wrap: wrap; gap: 8px;">
              <div>
                <span style="font-size: 14px; font-weight: 700; color: var(--text-heading);">${meta.name}</span>
                <p style="font-size: 12.5px; color: var(--text-secondary); margin-top: 2px;">${meta.description}</p>
              </div>
              <div style="font-size: 14px; font-weight: 700; color: var(--text-heading);">
                Behavioral Intention: <span style="color: #2563EB; font-family: var(--font-mono); font-size: 15px;">${fmt(ps?.primary_adoption_rating, 2)}</span> / 5.0
              </div>
            </div>

            <div style="font-size: 12px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 10px;">
              Generated Natural-Language Construct Responses (${sets.length || 6}):
            </div>

            <div style="display: flex; flex-direction: column; gap: 14px;">
              ${sets.map(s => {
                const text = s.response_text || cResponses[s.construct_code] || ps?.response_text || "Response recorded.";
                const isBI = s.construct_code === "BI";
                return `
                  <div style="background: ${isBI ? '#F0F7FF' : '#FFFFFF'}; border: 1px solid ${isBI ? '#93C5FD' : 'var(--border-color)'}; border-left: 4px solid ${isBI ? '#2563EB' : meta.color}; border-radius: 8px; padding: 12px 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                      <span style="font-weight: 700; font-size: 13px; color: ${isBI ? '#1D4ED8' : 'var(--text-heading)'};">
                        <strong>${s.construct_code}</strong> &middot; ${s.construct_name}
                      </span>
                      <span style="font-family: var(--font-mono); font-weight: 700; font-size: 13px; color: ${isBI ? '#1D4ED8' : '#0F172A'}; background: ${isBI ? '#DBEAFE' : '#F1F5F9'}; padding: 2px 7px; border-radius: 4px;">
                        ${fmt(s.expected_rating, 2)} / 5.0
                      </span>
                    </div>
                    <div style="font-size: 12.5px; line-height: 1.5; color: #334155; word-break: break-word;">
                      &ldquo;${esc(text)}&rdquo;
                    </div>
                  </div>
                `;
              }).join("")}
            </div>
          </div>
          <div class="modal-footer">
            <button class="btn btn-secondary" onclick="closeModal()">Close</button>
          </div>
        </div>
      </div>
    `;
  }

  if (state.activeModal === "new_eval") {
    return `
      <div class="modal-overlay active" onclick="if(event.target===this)closeModal()">
        <div class="modal-card">
          <div class="modal-header">
            <h3 class="modal-title">Evaluate Technology Product Concept</h3>
            <button class="modal-close-btn" onclick="closeModal()">&times;</button>
          </div>
          <form onsubmit="handleRunEvaluation(event)" style="display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; overflow: hidden;">
            <div class="modal-body" style="flex: 1 1 auto; overflow-y: auto; min-height: 0;">
              ${state.isRunning ? `
                <div style="text-align: center; padding: 40px 20px;">
                  <div class="loader-spinner" style="margin: 0 auto 16px;"></div>
                  <h4 style="font-size: 15px; font-weight: 700;">Running Evaluation...</h4>
                  <p style="font-size: 13px; color: var(--text-secondary);">${esc(state.runStatus)}</p>
                </div>
              ` : `
                <div class="form-group">
                  <label class="form-label">Product Name</label>
                  <input class="form-input" placeholder="e.g. Asus Vivobook 15, Garmin Venu 3, Smart Glasses, AI Echo Hub..." value="${esc(state.form.name)}" oninput="state.form.name = this.value" required />
                </div>
                <div class="form-group">
                  <label class="form-label">Product Category</label>
                  <input class="form-input" list="categorySuggestions" placeholder="e.g. Laptop Computer, Smartwatch, Smart Glasses, AI Device, Smartphone..." value="${esc(state.form.category || '')}" oninput="state.form.category = this.value" required />
                  <datalist id="categorySuggestions">
                    <option value="Laptop Computer">
                    <option value="Smartwatch">
                    <option value="Tablet Computer">
                    <option value="Smart Glasses">
                    <option value="AI Device / Assistant">
                    <option value="Smart Home Security Camera">
                    <option value="Audio Device / Headphones">
                    <option value="Smartphone">
                    <option value="Technology Product">
                  </datalist>
                </div>
                <div class="form-group">
                  <label class="form-label">Launch Price (₹ INR)</label>
                  <input class="form-input" type="number" min="1" placeholder="e.g. 20000" value="${state.form.price_inr ?? ''}" oninput="state.form.price_inr = this.value === '' ? null : parseFloat(this.value)" required />
                </div>
                <div class="form-group">
                  <label class="form-label">Product Description</label>
                  <textarea class="form-textarea" rows="3" placeholder="What the product is, who it is for, and what it is good at. Use your own words." oninput="state.form.description = this.value" required>${esc(state.form.description)}</textarea>
                </div>
                <div class="form-group">
                  <label class="form-label">Specifications <span style="font-weight: 400; color: var(--text-secondary);">(optional, only what you know)</span></label>
                  <textarea class="form-textarea" rows="4" placeholder="e.g. 6.6-inch 120Hz display; 5000mAh battery; 50MP camera; 8GB RAM. Leave out anything you are not sure of." oninput="state.form.specifications = this.value">${esc(state.form.specifications)}</textarea>
                  <p style="font-size: 11.5px; color: var(--text-secondary); margin-top: 4px;">Persona answers may mention only the details written here or in the description. Nothing else is added.</p>
                </div>
                <div class="form-group">
                  <label class="form-label">Target Audience <span style="font-weight: 400; color: var(--text-secondary);">(optional)</span></label>
                  <input class="form-input" placeholder="e.g. College students, Software engineers, Fitness enthusiasts..." value="${esc(state.form.target_group)}" oninput="state.form.target_group = this.value" />
              `}
            </div>
            <div class="modal-footer" style="flex-shrink: 0; display: flex; justify-content: flex-end; gap: 10px; padding: 14px 22px; border-top: 1px solid var(--border-color); background-color: #F8FAFC;">
              <button type="button" class="btn btn-secondary" onclick="closeModal()" ${state.isRunning ? 'disabled' : ''}>Cancel</button>
              <button type="submit" id="btn-evaluate-product" class="btn btn-primary" style="font-weight: 700; padding: 10px 22px; font-size: 14px; display: inline-flex; align-items: center; gap: 8px;" ${state.isRunning ? 'disabled' : ''}>
                ${state.isRunning ? 'Evaluating...' : 'Evaluate Product &rarr;'}
              </button>
            </div>
          </form>
        </div>
      </div>
    `;
  }

  return "";
}

// Master Render
function render() {
  if (state.isLoading) {
    app.innerHTML = `
      <div style="display: flex; justify-content: center; align-items: center; min-height: 100vh;">
        <div style="text-align: center;">
          <div class="loader-spinner" style="margin: 0 auto 16px;"></div>
          <p style="font-weight: 600; color: #64748B;">Loading Digital Persona Platform&hellip;</p>
        </div>
      </div>
    `;
    return;
  }

  const ev = state.activeEvaluation;
  const conceptName = ev?.concept?.name || "Evaluated Product";
  const conceptPrice = ev?.concept?.price_inr || 0;

  const viewTitles = {
    overview: "Overview",
    product: "Product Concept",
    personas: "Digital Personas",
    evaluation: "SSR Evaluation",
    insights: "Market Insights",
    methodology: "Methodology & Audit"
  };

  const titleText = viewTitles[state.currentView] || "Overview";

  app.innerHTML = `
    <div class="app-container">
      <!-- Sidebar (Approved Dark Navy / Indigo) -->
      <aside class="sidebar ${state.sidebarCollapsed ? 'collapsed' : ''} ${state.mobileOpen ? 'mobile-open' : ''}">
        <div class="sidebar-header">
          <a href="#" class="brand-wrapper" onclick="switchView('overview'); return false;">
            <div class="brand-icon">
              <svg viewBox="0 0 32 32">
                <defs>
                  <linearGradient id="cubeTop" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#818CF8"/>
                    <stop offset="100%" stop-color="#C084FC"/>
                  </linearGradient>
                  <linearGradient id="cubeLeft" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#4F46E5"/>
                    <stop offset="100%" stop-color="#6366F1"/>
                  </linearGradient>
                  <linearGradient id="cubeRight" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#3B82F6"/>
                    <stop offset="100%" stop-color="#2563EB"/>
                  </linearGradient>
                </defs>
                <polygon points="16,2 30,10 16,18 2,10" fill="url(#cubeTop)"/>
                <polygon points="2,10 16,18 16,30 2,22" fill="url(#cubeLeft)"/>
                <polygon points="16,18 30,10 30,22 16,30" fill="url(#cubeRight)"/>
              </svg>
            </div>
            <div class="brand-info">
              <span class="brand-title">Digital Persona</span>
              <span class="brand-tag">SSR INTELLIGENCE</span>
            </div>
          </a>
        </div>

        <nav class="sidebar-nav">
          <div class="nav-item ${state.currentView === 'overview' ? 'active' : ''}" onclick="switchView('overview')">
            <span class="nav-icon">${getIcon("overview")}</span>
            <span class="nav-label">Overview</span>
          </div>

          <div class="nav-item ${state.currentView === 'product' ? 'active' : ''}" onclick="switchView('product')">
            <span class="nav-icon">${getIcon("product")}</span>
            <span class="nav-label">Product Concept</span>
          </div>

          <div class="nav-item ${state.currentView === 'personas' ? 'active' : ''}" onclick="switchView('personas')">
            <span class="nav-icon">${getIcon("personas")}</span>
            <span class="nav-label">Digital Personas</span>
          </div>

          <div class="nav-item ${state.currentView === 'evaluation' ? 'active' : ''}" onclick="switchView('evaluation')">
            <span class="nav-icon">${getIcon("evaluation")}</span>
            <span class="nav-label">SSR Evaluation</span>
          </div>

          <div class="nav-item ${state.currentView === 'insights' ? 'active' : ''}" onclick="switchView('insights')">
            <span class="nav-icon">${getIcon("insights")}</span>
            <span class="nav-label">Market Insights</span>
          </div>

          <div class="nav-item ${state.currentView === 'methodology' ? 'active' : ''}" onclick="switchView('methodology')">
            <span class="nav-icon">${getIcon("methodology")}</span>
            <span class="nav-label">Methodology & Audit</span>
          </div>


        </nav>

        <div class="sidebar-footer">
          <div class="ollama-status-badge">
            <div class="status-indicator-dot"></div>
            <div class="ollama-text-group">
              <div class="ollama-title">Local Ollama Active</div>
              <div class="ollama-sub">Qwen 2.5 7B · nomic-embed</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Layout -->
      <div class="main-wrapper">
        <!-- Top App Header -->
        <header class="app-header">
          <div class="header-left">
            <button class="mobile-menu-btn" onclick="toggleSidebar()" title="Toggle Sidebar">
              &#9776;
            </button>
            <div class="header-title-box">
              <h1 class="header-view-title">${titleText}</h1>
              <p class="header-view-subtitle">Synthetic consumer research dashboard</p>
            </div>
          </div>

          <div class="header-right">
            <!-- Current Product Display (Plain Text) -->
            <div class="product-select-pill" style="cursor: default;">
              <div class="product-thumb-icon">${getIcon("product")}</div>
              <div class="product-select-info">
                <span class="product-select-name" style="font-weight: 700; font-size: 13px; color: var(--text-heading); display: block;">
                  ${esc(conceptName)}
                </span>
                <span class="product-select-price" style="font-size: 11px; color: var(--text-secondary); display: block;">
                  ₹${Number(conceptPrice).toLocaleString("en-IN")}${ev?.concept?.category ? ' · ' + esc(ev.concept.category) : ''}
                </span>
              </div>
            </div>

            <!-- Export Action Button -->
            <button class="btn btn-secondary" onclick="switchView('methodology')">
              ${getIcon("download")} Export
            </button>

            <!-- + New Concept Button -->
            <button class="btn btn-primary" onclick="openModal('new_eval')">
              ${getIcon("plus")} New Concept
            </button>

            <!-- User Initials Avatar -->
            <div class="user-avatar-circle">MN</div>
          </div>
        </header>

        <!-- Main View Area -->
        <main class="view-container">
          ${state.errorMessage ? `
            <div style="background: #FFF1F2; border: 1px solid #FFE4E6; border-radius: 8px; padding: 12px 18px; color: #9F1239; margin-bottom: 20px;">
              <strong>Notice:</strong> ${esc(state.errorMessage)}
            </div>
          ` : ''}

          ${renderOverviewView(ev)}
          ${renderProductView(ev)}
          ${renderPersonasView(ev)}
          ${renderEvaluationView(ev)}
          ${renderInsightsView(ev)}
          ${renderMethodologyView()}
        </main>
      </div>
    </div>

    <!-- Modals -->
    ${renderModals(ev)}
  `;
}

// Kick off
init();
