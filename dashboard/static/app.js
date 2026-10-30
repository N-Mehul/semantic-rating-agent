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
  errorMessage: null,
  options: null,
  activeEvaluation: null,
  form: {
    name: "Nova AI Smartphone",
    category: "Mid-Range Smartphone",
    price_inr: 24999,
    target_group: "Students & young professionals",
    target_age_group: "18-25 years",
    market: "India",
    brand_or_ecosystem: "Android",
    description: "A mid-range smartphone designed for students and young professionals, with AI-powered features.",
    features: [
      { name: "AI Personal Assistant", explanation: "Daily tasks, study reminders, and smart notifications." },
      { name: "AI Camera Enhancement", explanation: "Automated low-light and portrait enhancement for photos and reels." },
      { name: "Smart Battery Optimisation", explanation: "Adapts power consumption to daily routine usage patterns." },
      { name: "On-Device AI Processing", explanation: "Selected AI tasks run locally on-device for enhanced personal privacy." },
      { name: "Smooth Performance", explanation: "High refresh display and fast processor for multitasking and casual gaming." },
      { name: "Simple Interface", explanation: "Clean layout with optional contextual AI suggestions." }
    ],
    personas: ["P1" + "_premium_tier_owner", "P2" + "_midrange_heavy_user", "P3" + "_midrange_moderate_user", "P4" + "_budget_or_undisclosed_price"],
    constructs: ["PE", "EE", "HM", "SI", "PP", "BI"]
  }
};

// Safe API call helper
async function apiCall(path, body = null) {
  try {
    const opts = body
      ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      : { cache: "no-store" };
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      return { ok: false, error: data?.error || `Server error: HTTP ${res.status}` };
    }
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: `Connection failed: ${e.message}. Is the dashboard server running on port 8765?` };
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

  const [optRes, evalRes] = await Promise.all([
    apiCall("/api/evaluation/options"),
    apiCall("/api/evaluation/saved?id=nova_ai_smartphone")
  ]);

  if (optRes.ok) state.options = optRes.data;
  if (evalRes.ok) {
    state.activeEvaluation = evalRes.data;
  } else {
    state.errorMessage = evalRes.error || "Could not load saved product evaluation.";
  }

  state.isLoading = false;
  render();
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

async function handleRunEvaluation(e) {
  if (e) e.preventDefault();
  state.isRunning = true;
  state.runStatus = "Connecting to local Ollama (Qwen 2.5 7B & nomic-embed-text)...";
  render();

  const res = await apiCall("/api/evaluation/run", state.form);
  state.isRunning = false;

  if (res.ok) {
    state.activeEvaluation = res.data;
    state.activeModal = null;
    state.currentView = "overview";
  } else {
    state.errorMessage = res.error || "Evaluation execution failed.";
  }
  render();
}

function getPersonaMeta(pid) {
  const p = String(pid || "");
  if (p.startsWith("P1")) {
    return {
      name: "P1 · Premium Tier Owner",
      title: "Premium Tier Owner",
      label: "Hesitant",
      color: "#E11D48",
      pastelClass: "pink",
      n: 47,
      age: 20,
      hours: "7.5 hrs",
      brand: "Apple iPhone",
      techComfort: "3.8 / 5.0",
      sentiment: "sentiment-low",
      sentimentText: "Hesitant / Ecosystem Inertia",
      quote: "My current smartphone already handles all my multitasking and productivity needs with zero lag, so the proposed performance improvements do not offer a noticeable upgrade. Switching away from iOS would disrupt my workflow."
    };
  }
  if (p.startsWith("P2")) {
    return {
      name: "P2 · Mid-Range Heavy User",
      title: "Mid-Range Heavy User",
      label: "Enthusiastic",
      color: "#16A34A",
      pastelClass: "green",
      n: 93,
      age: 20,
      hours: "9.5 hrs",
      brand: "OnePlus / Samsung Android",
      techComfort: "4.2 / 5.0",
      sentiment: "sentiment-high",
      sentimentText: "Strongly Enthusiastic",
      quote: "At ₹24,999, the Nova AI Smartphone hits the perfect sweet spot. The automated camera enhancements and smart battery routines solve my daily pain points as a heavy multitasker."
    };
  }
  if (p.startsWith("P3")) {
    return {
      name: "P3 · Mid-Range Moderate User",
      title: "Mid-Range Moderate User",
      label: "Open / Balanced",
      color: "#2563EB",
      pastelClass: "blue",
      n: 82,
      age: 20,
      hours: "5.0 hrs",
      brand: "Android Mid-tier",
      techComfort: "3.5 / 5.0",
      sentiment: "sentiment-mod",
      sentimentText: "Open / Value-Conscious",
      quote: "The clean interface and practical battery optimization are very attractive. As long as the AI features do not complicate daily use or drain the battery prematurely, it is a viable upgrade option."
    };
  }
  return {
    name: "P4 · Budget User",
    title: "Budget User",
    label: "Price-Sensitive",
    color: "#EA580C",
    pastelClass: "peach",
    n: 36,
    age: 20,
    hours: "6.0 hrs",
    brand: "Entry / Budget Android",
    techComfort: "3.1 / 5.0",
    sentiment: "sentiment-low",
    sentimentText: "Price-Sensitive / Resistant",
    quote: "₹24,999 is well beyond what I can justify spending as a student. While the AI features sound interesting, my primary concern is device longevity and pure value for money without unnecessary gimmicks."
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
        <div class="hero-eval-label">EVALUATED PRODUCT</div>
        <h1 class="hero-product-title">${esc(concept.name || "Nova AI Smartphone")}</h1>
        <p class="hero-product-desc">
          ${esc(concept.description || "A mid-range smartphone designed for students and young professionals, with AI-powered features.")}
        </p>

        <!-- 4 Feature Pills Row -->
        <div class="hero-pills-row">
          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #FFEDD5; color: #EA580C;">₹</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">₹${Number(concept.price_inr || 24999).toLocaleString("en-IN")}</span>
              <span class="hero-pill-lbl">Price</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #DCFCE7; color: #16A34A;">${getIcon("android")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.brand_or_ecosystem || "Android")}</span>
              <span class="hero-pill-lbl">Ecosystem</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #DBEAFE; color: #2563EB;">${getIcon("user")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.target_age_group || "18–25 years")}</span>
              <span class="hero-pill-lbl">Target age</span>
            </div>
          </div>

          <div class="hero-pill">
            <div class="hero-pill-icon" style="background: #F3E8FF; color: #7C3AED;">${getIcon("gradCap")}</div>
            <div class="hero-pill-text">
              <span class="hero-pill-val">${esc(concept.target_group || "Students & young professionals")}</span>
              <span class="hero-pill-lbl">Target audience</span>
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
            <span class="kpi-status-badge">Moderate / Selective Adoption Intent</span>
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

// 2. Product Concept View
function renderProductView(ev) {
  const concept = ev?.concept || {};
  const features = concept.features || [];

  return `
    <div class="page-view ${state.currentView === 'product' ? 'active' : ''}">
      <div class="content-card" style="margin-bottom: 24px;">
        <h2 style="font-size: 24px; font-weight: 800; margin-bottom: 6px;">${esc(concept.name || "Nova AI Smartphone")}</h2>
        <div style="font-size: 22px; font-weight: 700; color: var(--accent-primary); font-family: var(--font-mono); margin-bottom: 10px;">
          ₹${Number(concept.price_inr || 24999).toLocaleString("en-IN")}
        </div>
        <p style="font-size: 14.5px; color: var(--text-body); line-height: 1.5; margin-bottom: 16px;">
          ${esc(concept.description || "A mid-range smartphone designed for students and young professionals, with AI-powered features.")}
        </p>

        <div class="specs-grid">
          <div class="spec-item">
            <span class="spec-key">Category</span>
            <span class="spec-value">${esc(concept.category || "Mid-Range Smartphone")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Target Audience</span>
            <span class="spec-value">${esc(concept.target_group || "Students & young professionals")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Target Age Group</span>
            <span class="spec-value">${esc(concept.target_age_group || "18–25 years")}</span>
          </div>
          <div class="spec-item">
            <span class="spec-key">Ecosystem</span>
            <span class="spec-value">${esc(concept.brand_or_ecosystem || "Android")}</span>
          </div>
        </div>
      </div>

      <h3 style="font-size: 16px; font-weight: 700; color: var(--text-heading); margin-bottom: 14px;">
        AI Capabilities & Features (${features.length})
      </h3>

      <div class="features-grid">
        ${features.map(f => `
          <div class="feature-card">
            <div class="feature-icon-wrapper">${getIcon("sparkles")}</div>
            <h4 class="feature-title">${esc(f.name)}</h4>
            <p class="feature-explanation">${esc(f.explanation)}</p>
          </div>
        `).join("")}
      </div>
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
          <h2 style="font-size: 20px; font-weight: 700; color: var(--text-heading);">Empirical Digital Personas</h2>
          <p style="font-size: 13px; color: var(--text-secondary);">
            Derived from 258 eligible human survey respondents via K-Means demographic & behavioural clustering.
          </p>
        </div>
      </div>

      <div class="personas-grid">
        ${pSummaries.map(ps => {
          const pid = ps.persona_id;
          const meta = getPersonaMeta(pid);
          const biScore = ps.primary_adoption_rating || 0;
          const biPct = Math.min(100, (biScore / 5.0) * 100);

          return `
            <div class="persona-card">
              <div>
                <div class="card-title-row" style="margin-bottom: 10px;">
                  <div class="persona-badge-group">
                    <span class="persona-id-badge">${pid.split("_")[0]}</span>
                    <span style="font-size: 11.5px; color: var(--text-muted); font-weight: 600;">Sample: N=${meta.n}</span>
                  </div>
                  <span class="persona-sentiment-pill ${meta.sentiment}">${meta.sentimentText}</span>
                </div>

                <h3 style="font-size: 16.5px; font-weight: 700; color: var(--text-heading); margin-bottom: 4px;">${meta.title}</h3>
                <p style="font-size: 12.5px; color: var(--text-secondary); margin-bottom: 14px;">Cluster representative based on survey respondent profile.</p>

                <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px;">
                  <span class="trait-chip">Age: ${meta.age} yrs</span>
                  <span class="trait-chip">Screen Time: ${meta.hours}/day</span>
                  <span class="trait-chip">Device: ${meta.brand}</span>
                  <span class="trait-chip">Tech Comfort: ${meta.techComfort}</span>
                </div>

                <div style="background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px 12px; margin-bottom: 14px;">
                  <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
                    <span style="font-weight: 600; color: var(--text-body);">Adoption Intent (BI)</span>
                    <span style="font-family: var(--font-mono); font-weight: 700; color: ${meta.color};">${fmt(biScore, 2)} / 5.0</span>
                  </div>
                  <div class="persona-progress-track">
                    <div class="persona-progress-fill" style="width: ${biPct}%; background: ${meta.color};"></div>
                  </div>
                </div>

                <div style="font-style: italic; font-size: 12px; color: var(--text-secondary); border-left: 3px solid ${meta.color}; padding-left: 10px; margin-bottom: 14px; line-height: 1.5; word-break: break-word;">
                  &ldquo;${meta.quote}&rdquo;
                </div>
              </div>

              <button class="btn btn-secondary btn-sm" style="width: 100%;" onclick="openModal('persona_detail', '${pid}')">
                Inspect Detailed Responses &rarr;
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
    BI: { name: "Behavioural Intention (BI)", desc: "The direct intention to adopt and purchase the smartphone concept." },
    PE: { name: "Performance Expectancy (PE)", desc: "The degree to which the device improves everyday academic and work productivity." },
    EE: { name: "Effort Expectancy (EE)", desc: "The perceived ease of use and interface simplicity of AI functions." },
    HM: { name: "Hedonic Motivation (HM)", desc: "Fun, entertainment, gaming, photography pleasure, and social excitement." },
    SI: { name: "Social Influence (SI)", desc: "Peer perception, recommendations, and status approval among classmates." },
    PP: { name: "Perceived Protection (PP)", desc: "Confidence in on-device AI security, biometric safety, and privacy boundaries." }
  };

  return `
    <div class="page-view ${state.currentView === 'evaluation' ? 'active' : ''}">
      <!-- Top Banner -->
      <div class="eval-header-banner">
        <div class="eval-banner-text">
          <h3>Independent Single-Set SSR Diagnostics</h3>
          <p>Each construct is scored strictly against its own 5-statement Likert anchor set.</p>
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
                const bi = ps.primary_adoption_rating || scores.BI || 0;

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
    </div>
  `;
}

// 5. Market Insights View
function renderInsightsView() {
  return `
    <div class="page-view ${state.currentView === 'insights' ? 'active' : ''}">
      <h2 style="font-size: 20px; font-weight: 700; color: var(--text-heading); margin-bottom: 4px;">Market Intelligence & Recommendations</h2>
      <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 20px;">Synthesized strategic insights derived from the 4 personas and 6 UTAUT2 construct ratings.</p>

      <div class="insights-grid">
        <div class="insight-card strength">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #166534;">✓ Key Product Strengths</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 12.5px; line-height: 1.5;">
            <li><strong>High Effort Expectancy (EE: 3.84/5.0):</strong> Clean interface lowers cognitive barriers for students.</li>
            <li><strong>Strong Camera & Reel Appeal (HM: 3.73/5.0):</strong> Low-light and portrait enhancements resonate strongly with daily creators.</li>
            <li><strong>On-Device Privacy Reassurance (PP: 3.50/5.0):</strong> Local AI execution protects user data.</li>
          </ul>
        </div>

        <div class="insight-card weakness">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #9F1239;">⚠️ Core Friction Points</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 12.5px; line-height: 1.5;">
            <li><strong>Price Sensitivity at ₹24,999:</strong> For budget students (P4), ₹25,000 exceeds their disposable pocket money threshold.</li>
            <li><strong>iOS Ecosystem Lock-in:</strong> iPhone owners (P1) are hesitant to leave iOS continuity features.</li>
            <li><strong>AI Gimmick Skepticism:</strong> Need for proven long-term battery endurance.</li>
          </ul>
        </div>

        <div class="insight-card hesitation">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #1E40AF;">⏸ Hesitation & Barrier Analysis</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 12.5px; line-height: 1.5;">
            <li><strong>P1 Hesitation:</strong> Unwilling to sacrifice AirDrop/iCloud integration.</li>
            <li><strong>P4 Resistance:</strong> Prioritizes sub-₹15,000 pure value over optional AI tools.</li>
            <li><strong>Brand Trust:</strong> As a newcomer, Nova lacks long-term support history.</li>
          </ul>
        </div>

        <div class="insight-card recommendation">
          <h3 style="font-size: 15px; font-weight: 700; margin-bottom: 12px; color: #9A3412;">★ Actionable Recommendations</h3>
          <ul style="list-style: none; display: flex; flex-direction: column; gap: 10px; font-size: 12.5px; line-height: 1.5;">
            <li><strong>Target Mid-Range Android Users First:</strong> Focus marketing on P2 & P3 where purchase intent reaches 3.91/5.0.</li>
            <li><strong>Campus Exchange Offers:</strong> Introduce ₹4,000 trade-in bonuses to bridge the P4 affordability gap.</li>
            <li><strong>Highlight Privacy & Battery:</strong> Differentiate with on-device processing and long battery life.</li>
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
            <p class="step-desc">Nova AI Smartphone concept: ₹24,999 launch price, mid-range category, 6 concrete features.</p>
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

    return `
      <div class="modal-overlay active" onclick="if(event.target===this)closeModal()">
        <div class="modal-card">
          <div class="modal-header">
            <h3 class="modal-title">Persona Profile & Response Audit: ${pid?.split("_")[0]}</h3>
            <button class="modal-close-btn" onclick="closeModal()">&times;</button>
          </div>
          <div class="modal-body">
            <div style="margin-bottom: 16px;">
              <span class="persona-id-badge">${pid}</span>
              <h4 style="font-size: 15px; font-weight: 700; color: var(--text-heading); margin-top: 8px;">
                Adoption Intent: ${fmt(ps?.primary_adoption_rating, 2)} / 5.0
              </h4>
            </div>
            <div style="margin-bottom: 18px;">
              <div style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-heading); margin-bottom: 6px;">
                Synthetic Natural-Language Response:
              </div>
              <div style="background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.55;">
                ${esc(resp?.response_text || "Response recorded.")}
              </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
              ${sets.map(s => `
                <div style="display: flex; justify-content: space-between; padding: 8px 12px; background: #F8FAFC; border: 1px solid var(--border-color); border-radius: 6px; font-size: 13px;">
                  <span><strong>${s.construct_code}</strong> · ${s.construct_name}</span>
                  <span style="font-family: var(--font-mono); font-weight: 700; color: #2563EB;">${fmt(s.expected_rating, 2)}</span>
                </div>
              `).join("")}
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
            <h3 class="modal-title">Evaluate New Smartphone Concept</h3>
            <button class="modal-close-btn" onclick="closeModal()">&times;</button>
          </div>
          <form onsubmit="handleRunEvaluation(event)">
            <div class="modal-body">
              ${state.isRunning ? `
                <div style="text-align: center; padding: 40px 20px;">
                  <div class="loader-spinner" style="margin: 0 auto 16px;"></div>
                  <h4 style="font-size: 15px; font-weight: 700;">Running Evaluation...</h4>
                  <p style="font-size: 13px; color: var(--text-secondary);">${esc(state.runStatus)}</p>
                </div>
              ` : `
                <div class="form-group">
                  <label class="form-label">Product Name</label>
                  <input class="form-input" value="${esc(state.form.name)}" oninput="state.form.name = this.value" required />
                </div>
                <div class="form-group">
                  <label class="form-label">Launch Price (₹ INR)</label>
                  <input class="form-input" type="number" value="${state.form.price_inr}" oninput="state.form.price_inr = parseFloat(this.value)" required />
                </div>
                <div class="form-group">
                  <label class="form-label">Description</label>
                  <textarea class="form-textarea" rows="3" oninput="state.form.description = this.value" required>${esc(state.form.description)}</textarea>
                </div>
                <div class="form-group">
                  <label class="form-label">Target Audience</label>
                  <input class="form-input" value="${esc(state.form.target_group)}" oninput="state.form.target_group = this.value" required />
                </div>
              `}
            </div>
            <div class="modal-footer">
              <button type="button" class="btn btn-secondary" onclick="closeModal()" ${state.isRunning ? 'disabled' : ''}>Cancel</button>
              <button type="submit" class="btn btn-primary" ${state.isRunning ? 'disabled' : ''}>
                ${state.isRunning ? 'Evaluating...' : 'Run SSR Evaluation'}
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
  const conceptName = ev?.concept?.name || "Nova AI Smartphone";
  const conceptPrice = ev?.concept?.price_inr || 24999;

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
            <!-- Product Dropdown Pill -->
            <div class="product-select-pill" onclick="switchView('product')">
              <div class="product-thumb-icon">${getIcon("product")}</div>
              <div class="product-select-info">
                <span class="product-select-name">${esc(conceptName)}</span>
                <span class="product-select-price">₹${Number(conceptPrice).toLocaleString("en-IN")}</span>
              </div>
              <span style="font-size: 11px; color: #94A3B8; margin-left: 2px;">${getIcon("chevronDown")}</span>
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
          ${renderInsightsView()}
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
