// MotionScope frontend: no build step, talks to the FastAPI backend under /api.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const fmtInt = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString());
const fmtTime = (s) => {
  if (s === null || s === undefined) return "–";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s % 60)).padStart(2, "0")} min`;
};
const dirColor = (deg) => `hsl(${Math.round(deg)}, 82%, 62%)`;   // same hue wheel as the video overlay

const state = {
  health: null,
  video: null,
  job: null,
  report: null,
  calibration: null,
  pollTimer: null,
};

// ------------------------------------------------------------------ API
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  const text = await res.text();
  try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text }; }
  if (!res.ok) {
    const msg = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `Request failed (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return body;
}

function toast(msg, ms = 3200) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

// --------------------------------------------------------------- routing
function route() {
  const h = location.hash;
  const m = h.match(/^#\/v\/([0-9a-f]{32})(?:\/j\/([0-9a-f]{32}))?$/);
  if (m) {
    showView("workspace");
    openWorkspace(m[1], m[2]);
  } else {
    stopPolling();
    showView("landing");
    if (h && !h.startsWith("#/")) {
      const el = document.getElementById(h.slice(1));
      if (el) requestAnimationFrame(() => el.scrollIntoView());
    }
  }
}
function showView(name) {
  if (name === "workspace" && $("#view-workspace").hidden) scrollTo(0, 0);
  $("#view-landing").hidden = name !== "landing";
  $("#view-workspace").hidden = name !== "workspace";
  $(".nav-cta-label").textContent = name === "workspace" ? "New video" : "Analyse video";
  heroPlayback(name === "landing");
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- upload
const fileInput = $("#file-input");
const dz = $("#dropzone");

// One upload entry point in the nav. On the landing page it brings the
// dropzone into view and opens the picker; in the workspace it starts over.
$('[data-action="nav-cta"]').addEventListener("click", () => {
  if ($("#view-landing").hidden) location.hash = "#/";
  requestAnimationFrame(() => {
    dz.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    dz.focus({ preventScroll: true });
    fileInput.click();
  });
});
dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => {
  e.preventDefault();
  if (ev === "dragleave" && dz.contains(e.relatedTarget)) return;   // moving between children
  dz.classList.remove("drag");
}));
const dzTitle = $(".dz-title");
const DZ_TITLE = dzTitle.textContent;
new MutationObserver(() => { dzTitle.textContent = dz.classList.contains("drag") ? "Release to upload" : DZ_TITLE; })
  .observe(dz, { attributes: true, attributeFilter: ["class"] });
dz.addEventListener("drop", (e) => { const f = e.dataTransfer?.files?.[0]; if (f) upload(f); });
fileInput.addEventListener("change", () => { const f = fileInput.files[0]; if (f) upload(f); fileInput.value = ""; });

function uploadError(msg) {
  const el = $("#upload-error");
  el.textContent = msg;
  el.hidden = !msg;
}

function upload(file) {
  uploadError("");
  const exts = state.health?.allowed_extensions || [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"];
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (!exts.includes(ext)) return uploadError(`“${file.name}” is not a supported video type (${exts.join(", ")}).`);
  const maxMb = state.health?.max_upload_mb || 1024;
  if (file.size > maxMb * 1024 * 1024) return uploadError(`File is ${(file.size / 1048576).toFixed(0)} MB; the limit is ${maxMb} MB.`);
  if (file.size === 0) return uploadError("That file is empty.");

  const prog = $(".dz-progress", dz);
  const bar = $(".bar span", prog);
  const label = $(".dz-progress-label", prog);
  dz.classList.add("busy");
  prog.hidden = false;
  bar.style.transform = "scaleX(0)";
  label.textContent = "Uploading…";

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/videos");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const p = (e.loaded / e.total) * 100;
    bar.style.transform = `scaleX(${p / 100})`;
    label.textContent = p >= 100 ? "Validating video…" : `Uploading… ${p.toFixed(0)}%`;
  };
  xhr.onload = () => {
    dz.classList.remove("busy");
    prog.hidden = true;
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status >= 200 && xhr.status < 300) {
      state.calibration = null;
      location.hash = `#/v/${body.id}`;
    } else {
      uploadError(body.detail || `Upload failed (${xhr.status}).`);
    }
  };
  xhr.onerror = () => { dz.classList.remove("busy"); prog.hidden = true; uploadError("Network error while uploading. Is the server running?"); };
  const fd = new FormData();
  fd.append("file", file);
  xhr.send(fd);
}

// ----------------------------------------------------------------- landing
async function loadHealth() {
  try {
    state.health = await api("/api/health");
    $$('[data-bind="max-mb"]').forEach((e) => (e.textContent = state.health.max_upload_mb));
    $$('[data-bind="max-min"]').forEach((e) => (e.textContent = Math.round(state.health.max_duration_s / 60)));
    const h = state.health;
    if (h.demo_note) { const n = $("#demo-note"); n.textContent = h.demo_note; n.hidden = false; }
    if (h.max_process_size) {
      $$('input[name="process_size"]').forEach((r) => { r.disabled = +r.value > h.max_process_size; });
    }
    if (h.max_frames) $('input[name="max_frames"]').placeholder = `All (max ${h.max_frames})`;
    if (!state.health.yolox_model) {
      const opt = $('select[name="detector"] option[value="auto"]');
      opt.textContent = "Auto (motion blobs; YOLOX not installed)";
    }
  } catch { /* backend offline: upload will report */ }
}

async function loadDemos() {
  const grid = $("#demo-grid");
  let demos = [];
  try { demos = (await api("/api/demos")).demos || []; } catch { /* ignore */ }
  if (!demos.length) {
    grid.innerHTML = `<p class="section-sub">No demonstrations generated yet. Run <code>python scripts/make_demos.py</code>.</p>`;
    $("#hero-caption").textContent = "Upload a video to see the processed output here.";
    return;
  }
  grid.innerHTML = demos.map(demoCard).join("");
  $$(".demo-card", grid).forEach((card) => {
    const video = $("video", card);
    $$(".demo-switch button", card).forEach((b) => b.addEventListener("click", () => {
      const t = video.currentTime;
      const playing = !video.paused;
      $$(".demo-switch button", card).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      video.src = b.dataset.src;
      video.addEventListener("loadedmetadata", () => { video.currentTime = Math.min(t, video.duration || t); if (playing) video.play().catch(() => {}); }, { once: true });
    }));
  });

  const hero = demos.find((d) => d.slug === "vehicle-tracking") || demos[0];
  heroV.poster = heroO.poster = hero.files["poster.jpg"];
  heroV.src = hero.files["annotated.mp4"];
  heroO.src = hero.files["original.mp4"];
  heroPlayback(!$("#view-landing").hidden);
  const sp = hero.stats.speed;
  $("#hero-caption").textContent = `Real output: ${hero.stats.objects} vehicles tracked in ${fmt(hero.clip.seconds, 0)} s of ${hero.source}`
    + (sp?.median_kmh ? `, median ≈${fmt(sp.median_kmh, 0)} km/h under an assumed calibration.` : ".");
}

// ---- hero before/after wipe (signature interaction)
const heroWipe = $("#hero-wipe");
const heroV = $("#hero-video");
const heroO = $("#hero-original");
const heroRange = $("#hero-wipe-range");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
function setHeroWipe(pct) {
  pct = Math.max(0, Math.min(100, pct));
  heroWipe.style.setProperty("--wipe", `${pct}%`);
  heroRange.value = String(Math.round(pct));
}
function heroPlayback(on) {
  if (!heroV.src) return;
  if (on && !reduceMotion.matches) {
    heroO.currentTime = heroV.currentTime;
    heroV.play().catch(() => {});
    heroO.play().catch(() => {});
  } else { heroV.pause(); heroO.pause(); }
}
heroV.addEventListener("timeupdate", () => {
  if (Math.abs(heroO.currentTime - heroV.currentTime) > 0.12) heroO.currentTime = heroV.currentTime;
});
heroRange.addEventListener("input", () => setHeroWipe(+heroRange.value));
let wipeDrag = null;
heroWipe.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  wipeDrag = { id: e.pointerId, x: e.clientX, moved: false };
  heroWipe.setPointerCapture(e.pointerId);
  heroWipe.classList.add("dragging");
  if (e.pointerType === "mouse") moveWipe(e);
});
function moveWipe(e) {
  const r = heroWipe.getBoundingClientRect();
  setHeroWipe(((e.clientX - r.left) / r.width) * 100);
}
heroWipe.addEventListener("pointermove", (e) => {
  if (!wipeDrag || e.pointerId !== wipeDrag.id) return;
  wipeDrag.moved = true;
  moveWipe(e);
});
const endWipe = () => { wipeDrag = null; heroWipe.classList.remove("dragging"); };
heroWipe.addEventListener("pointerup", endWipe);
heroWipe.addEventListener("pointercancel", endWipe);   // vertical scroll on touch hands the gesture back

function demoCard(d) {
  const s = d.stats;
  const speed = s.speed?.median_kmh
    ? `<div><dt>median speed</dt><dd>≈${fmt(s.speed.median_kmh, 0)} km/h est.</dd></div>`
    : `<div><dt>camera shift</dt><dd>${fmt(s.camera_shift_px, 2)} px</dd></div>`;
  return `
  <article class="demo-card${d.orientation === "portrait" ? " feature" : ""}">
    <div class="demo-media">
      <div class="demo-switch" role="group" aria-label="Choose video">
        <button aria-pressed="true" data-src="${esc(d.files["annotated.mp4"])}">Processed</button>
        <button aria-pressed="false" data-src="${esc(d.files["original.mp4"])}">Original</button>
      </div>
      <video controls muted playsinline preload="none" poster="${esc(d.files["poster.jpg"])}" src="${esc(d.files["annotated.mp4"])}" aria-label="${esc(d.title)}"></video>
    </div>
    <div class="demo-body">
      <h3>${esc(d.title)}</h3>
      <p>${esc(d.summary)}</p>
      <dl class="demo-meta mono">
        <div><dt>source</dt><dd>${esc(d.source)} · ${esc(s.resolution)} · ${fmt(d.clip.seconds, 0)} s</dd></div>
        <div><dt>features</dt><dd>${fmtInt(s.unique_features)}</dd></div>
        <div><dt>${s.detector === "yolox" ? "vehicles" : "objects"}</dt><dd>${fmtInt(s.objects)}</dd></div>
        ${speed}
      </dl>
      ${d.calibration_note ? `<p class="demo-note">${esc(d.calibration_note)}</p>` : ""}
      <div class="demo-links">
        <a href="${esc(d.files["report.json"])}" target="_blank" rel="noopener">report.json</a>
        <a href="${esc(d.files["tracks.csv"])}" download>tracks.csv</a>
        <a href="${esc(d.files["trajectories.png"])}" target="_blank" rel="noopener">trajectories</a>
      </div>
    </div>
  </article>`;
}

// --------------------------------------------------------------- workspace
async function openWorkspace(videoId, jobId) {
  if (state.video?.id === videoId && (!jobId || state.job?.id === jobId)) return;
  stopPolling();
  resetResults();
  let meta;
  try {
    meta = await api(`/api/videos/${videoId}`);
  } catch (e) {
    toast(e.status === 404 ? "That video is no longer on the server." : e.message);
    location.hash = "#/";
    return;
  }
  if (state.video?.id !== videoId) state.calibration = null;
  state.video = meta;
  $("#ws-title").textContent = meta.filename;
  const i = meta.info;
  $("#ws-meta").innerHTML = [
    ["Resolution", `${i.width}×${i.height}`],
    ["Duration", fmtTime(i.duration_s)],
    ["Frame rate", `${fmt(i.fps, 2)} fps`],
    ["Frames", fmtInt(i.frame_count)],
    ["Codec", esc(i.codec)],
    ["Size", `${(meta.size_bytes / 1048576).toFixed(1)} MB`],
  ].map(([k, v]) => `<div><dt>${k}</dt><dd title="${v}">${v}</dd></div>`).join("");
  $('input[name="max_frames"]').max = i.frame_count;
  setStageSources();
  updateCalibrationStatus();
  setTab("original");
  $("#settings").hidden = false;
  $("#job").hidden = true;
  if (jobId) attachJob(jobId);
}

function setStageSources() {
  const v = state.video;
  const vo = $("#v-original");
  vo.poster = `/api/videos/${v.id}/poster`;
  if (v.preview === "ready") {
    vo.src = `/api/videos/${v.id}/stream`;
    $("#c-original").src = vo.src;
    $("#preview-pending").hidden = true;
  } else {
    vo.removeAttribute("src");
    $("#preview-pending").hidden = false;
    $("#preview-pending").textContent = v.preview === "failed"
      ? "This format can't be previewed in the browser, but it can still be analysed."
      : "Preparing a browser-playable preview of this format…";
    if (v.preview === "pending") setTimeout(async () => {
      try { const m = await api(`/api/videos/${v.id}`); if (state.video?.id === m.id) { state.video = m; setStageSources(); } } catch { /* ignore */ }
    }, 2500);
  }
}

// tabs: segmented control with a sliding indicator
const segTabs = $(".stage-bar .seg");
$$("button", segTabs).forEach((b) => b.addEventListener("click", () => setTab(b.dataset.view)));
segTabs.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const tabs = $$("button:not(:disabled)", segTabs);
  const i = tabs.findIndex((b) => b.getAttribute("aria-selected") === "true");
  const next = tabs[(i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
  setTab(next.dataset.view); next.focus();
});
function moveIndicator() {
  const active = $('button[aria-selected="true"]', segTabs);
  if (!active || !active.offsetWidth) return;
  const ind = $(".seg-indicator", segTabs);
  ind.style.width = `${active.offsetWidth}px`;
  ind.style.transform = `translateX(${active.offsetLeft}px)`;
}
// Re-measure whenever the control changes size (viewport, web-font swap, tabs enabling).
new ResizeObserver(moveIndicator).observe(segTabs);
document.fonts?.ready.then(moveIndicator);
function setTab(view) {
  $$("button", segTabs).forEach((b) => {
    const on = b.dataset.view === view;
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  });
  $("#stage").dataset.mode = view;
  $$("#stage video").forEach((v) => v.pause());
  $(".compare").hidden = view !== "compare";
  requestAnimationFrame(moveIndicator);
}

// compare
const cO = $("#c-original");
const cP = $("#c-processed");
const cmpPlay = $("#compare-play");
const cmpSeek = $("#compare-seek");
$("#compare-wipe").addEventListener("input", (e) => $(".compare-inner").style.setProperty("--wipe", `${e.target.value}%`));
cmpPlay.addEventListener("click", () => {
  if (cP.paused) { cO.currentTime = cP.currentTime; Promise.all([cP.play(), cO.play()]).catch(() => {}); cmpPlay.textContent = "Pause"; }
  else { cP.pause(); cO.pause(); cmpPlay.textContent = "Play"; }
});
cmpSeek.addEventListener("input", () => {
  if (!cP.duration) return;
  const t = (cmpSeek.value / 1000) * cP.duration;
  cP.currentTime = t; cO.currentTime = t;
});
cP.addEventListener("timeupdate", () => {
  if (cP.duration) cmpSeek.value = String(Math.round((cP.currentTime / cP.duration) * 1000));
  if (!cP.paused && Math.abs(cO.currentTime - cP.currentTime) > 0.12) cO.currentTime = cP.currentTime;
});
cP.addEventListener("ended", () => { cO.pause(); cmpPlay.textContent = "Play"; });

// ------------------------------------------------------------- calibration
const calDialog = $("#cal-dialog");
const calCanvas = $("#cal-canvas");
const calCtx = calCanvas.getContext("2d");
const calImg = new Image();
let calPts = [];
let dragIdx = -1;

$('[data-action="open-calibration"]').addEventListener("click", () => {
  calPts = state.calibration ? state.calibration.points.map((p) => [...p]) : [];
  const f = $("#cal-form");
  f.width_m.value = state.calibration?.width_m ?? "";
  f.length_m.value = state.calibration?.length_m ?? "";
  f.label.value = state.calibration?.label ?? "";
  $("#cal-error").hidden = true;
  calImg.onload = () => { calCanvas.width = calImg.naturalWidth; calCanvas.height = calImg.naturalHeight; drawCal(); };
  calImg.src = `/api/videos/${state.video.id}/poster`;
  calDialog.showModal();
});
$('[data-action="close-calibration"]').addEventListener("click", () => calDialog.close());
$('[data-action="reset-calibration"]').addEventListener("click", () => { calPts = []; drawCal(); });
$('[data-action="clear-calibration"]').addEventListener("click", () => { state.calibration = null; updateCalibrationStatus(); });

function canvasPoint(e) {
  const r = calCanvas.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
}
calCanvas.addEventListener("pointerdown", (e) => {
  const p = canvasPoint(e);
  const r = calCanvas.getBoundingClientRect();
  dragIdx = calPts.findIndex((q) => Math.hypot((q[0] - p[0]) * r.width, (q[1] - p[1]) * r.height) < 14);
  if (dragIdx < 0 && calPts.length < 4) { calPts.push(p); dragIdx = calPts.length - 1; }
  calCanvas.setPointerCapture(e.pointerId);
  drawCal();
});
calCanvas.addEventListener("pointermove", (e) => { if (dragIdx >= 0) { calPts[dragIdx] = canvasPoint(e); drawCal(); } });
calCanvas.addEventListener("pointerup", () => { dragIdx = -1; });

function drawCal() {
  const W = calCanvas.width, H = calCanvas.height;
  if (!W) return;
  calCtx.drawImage(calImg, 0, 0, W, H);
  const s = W / 900;
  const px = calPts.map(([x, y]) => [x * W, y * H]);
  if (px.length > 1) {
    calCtx.beginPath();
    px.forEach(([x, y], i) => (i ? calCtx.lineTo(x, y) : calCtx.moveTo(x, y)));
    if (px.length === 4) { calCtx.closePath(); calCtx.fillStyle = "rgba(64,214,230,0.16)"; calCtx.fill(); }
    calCtx.strokeStyle = "#40d6e6"; calCtx.lineWidth = 2 * s; calCtx.stroke();
  }
  calCtx.font = `${600} ${14 * s}px "JetBrains Mono", monospace`;
  px.forEach(([x, y], i) => {
    calCtx.beginPath(); calCtx.arc(x, y, 6 * s, 0, Math.PI * 2);
    calCtx.fillStyle = "#40d6e6"; calCtx.fill();
    calCtx.lineWidth = 2 * s; calCtx.strokeStyle = "#0d0e10"; calCtx.stroke();
    calCtx.fillStyle = "#fff"; calCtx.fillText("ABCD"[i], x + 9 * s, y - 9 * s);
  });
  if (px.length >= 2) edgeLabel(px[0], px[1], "width", s);
  if (px.length >= 3) edgeLabel(px[1], px[2], "length", s);
  $("#cal-count").textContent = `${calPts.length} / 4 points`;
}
function edgeLabel(a, b, text, s) {
  calCtx.fillStyle = "#ffb228";
  calCtx.font = `${12 * s}px "JetBrains Mono", monospace`;
  calCtx.fillText(text, (a[0] + b[0]) / 2 + 8 * s, (a[1] + b[1]) / 2 + 16 * s);
}
function isConvex(q) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
    const cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cr) < 1e-9) return false;
    if (sign && Math.sign(cr) !== sign) return false;
    sign = Math.sign(cr);
  }
  return true;
}
$("#cal-form").addEventListener("submit", (e) => {
  const f = e.currentTarget;
  const err = (m) => { e.preventDefault(); $("#cal-error").textContent = m; $("#cal-error").hidden = false; };
  if (calPts.length !== 4) return err("Place all four points (A, B, C, D).");
  if (!isConvex(calPts)) return err("Points must go around the rectangle in order (A→B→C→D) without crossing.");
  const w = parseFloat(f.width_m.value), l = parseFloat(f.length_m.value);
  if (!(w >= 0.1 && w <= 500) || !(l >= 0.1 && l <= 500)) return err("Enter both real-world distances in metres (0.1–500).");
  state.calibration = { points: calPts.map(([x, y]) => [+x.toFixed(5), +y.toFixed(5)]), width_m: w, length_m: l, label: f.label.value.trim() };
  updateCalibrationStatus();
});
function updateCalibrationStatus() {
  const sum = $("#cal-summary");
  sum.textContent = state.calibration ? `${state.calibration.width_m} × ${state.calibration.length_m} m` : "Off";
  sum.classList.toggle("ok", !!state.calibration);
  if (state.calibration) $("#cal-group").open = true;
  const c = state.calibration;
  const el = $("#cal-status");
  el.classList.toggle("ok", !!c);
  el.textContent = c
    ? `Calibrated: ${c.width_m} m × ${c.length_m} m reference${c.label ? ` (${c.label})` : ""}. Speeds will be reported as estimates.`
    : "Not calibrated. Results stay in pixels; no km/h is reported.";
  $('[data-action="clear-calibration"]').hidden = !c;
}

// --------------------------------------------------------------------- jobs
$("#settings").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.currentTarget;
  const errEl = $("#job-error");
  errEl.hidden = true;
  const body = {
    video_id: state.video.id,
    process_size: +f.process_size.value,
    detector: f.detector.value,
    compensate_camera: f.compensate_camera.checked,
    layers: {
      flow_vectors: f.layer_flow_vectors.checked,
      feature_trails: f.layer_feature_trails.checked,
      objects: f.layer_objects.checked,
      saliency: f.layer_saliency.checked,
      hud: f.layer_hud.checked,
      calibration: true,
    },
  };
  if (f.fps_override.value) body.fps_override = +f.fps_override.value;
  if (f.max_frames.value) body.max_frames = +f.max_frames.value;
  if (state.calibration) body.calibration = state.calibration;
  $("#start-btn").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    resetResults();
    history.replaceState(null, "", `#/v/${state.video.id}/j/${job.id}`);
    attachJob(job.id, job);
  } catch (err) {
    errEl.textContent = err.message;
    errEl.hidden = false;
  } finally {
    $("#start-btn").disabled = false;
  }
});

function attachJob(jobId, initial) {
  state.job = initial || { id: jobId, status: "queued" };
  $("#settings").hidden = true;
  $("#job").hidden = false;
  renderJob(state.job);
  stopPolling();
  poll();
}
async function poll() {
  try {
    const j = await api(`/api/jobs/${state.job.id}`);
    state.job = j;
    renderJob(j);
    if (j.status === "completed") return onCompleted(j);
    if (j.status === "failed" || j.status === "cancelled") return;
  } catch (e) {
    if (e.status === 404) { toast("That job no longer exists."); return showSettings(); }
  }
  state.pollTimer = setTimeout(poll, 600);
}
function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; }

const STATUS_TEXT = { queued: "Queued", running: "Processing", paused: "Paused", completed: "Completed", failed: "Failed", cancelled: "Cancelled" };
function renderJob(j) {
  const box = $("#job");
  box.dataset.status = j.status;
  $("#job-state").textContent = STATUS_TEXT[j.status] || j.status;
  const pct = j.status === "completed" ? 100 : (j.progress || 0) * 100;
  $("#job-pct").textContent = `${pct.toFixed(0)}%`;
  $("#job-bar").style.transform = `scaleX(${pct / 100})`;
  $("#job-frames").textContent = `${fmtInt(j.frames_done || 0)} / ${fmtInt(j.frames_total || 0)}`;
  $("#job-fps").textContent = j.throughput_fps ? `${fmt(j.throughput_fps)} fps` : "–";
  $("#job-eta").textContent = j.eta_s != null ? fmtTime(j.eta_s) : (j.status === "completed" ? "done" : "–");
  const active = ["queued", "running", "paused"].includes(j.status);
  $('[data-action="pause"]').hidden = j.status !== "running";
  $('[data-action="resume"]').hidden = j.status !== "paused";
  $('[data-action="cancel"]').hidden = !active;
  $('[data-action="rerun"]').hidden = active;
  $('[data-action="view-results"]').hidden = j.status !== "completed";
  $("#jump-results").hidden = j.status !== "completed";
  const err = $("#job-error");
  if (j.status === "failed") { err.textContent = j.error || "Processing failed."; err.hidden = false; }
}
for (const action of ["pause", "resume", "cancel"]) {
  $(`[data-action="${action}"]`).addEventListener("click", async () => {
    try { renderJob((state.job = await api(`/api/jobs/${state.job.id}/${action}`, { method: "POST" }))); }
    catch (e) { toast(e.message); }
  });
}
$('[data-action="rerun"]').addEventListener("click", showSettings);
function showSettings() {
  stopPolling();
  $("#settings").hidden = false;
  $("#job").hidden = true;
  if (state.job?.status !== "failed") $("#job-error").hidden = true;
  $("#jump-results").hidden = $("#results").hidden;
  history.replaceState(null, "", `#/v/${state.video.id}`);
}
// In-page links to the results must not change the hash (the hash is the router).
$$('a[href="#results"]').forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault();
  $("#results").scrollIntoView({ behavior: reduceMotion.matches ? "auto" : "smooth", block: "start" });
}));

async function onCompleted(j) {
  const bust = `?t=${j.finished_at}`;
  const vp = $("#v-processed");
  vp.src = j.files["annotated.mp4"] + bust;
  // Frame 0 has no predecessor and therefore no overlay; open on a tracked frame.
  vp.addEventListener("loadedmetadata", () => { if (vp.currentTime === 0) vp.currentTime = Math.min(1, vp.duration / 3); }, { once: true });
  cP.src = j.files["annotated.mp4"] + bust;
  if (state.video.preview === "ready") cO.src = `/api/videos/${state.video.id}/stream`;
  $("#tab-processed").disabled = false;
  $("#tab-compare").disabled = state.video.preview !== "ready";
  setTab("processed");
  try {
    state.report = await api(`/api/jobs/${j.id}/report`);
    renderResults(state.report, j);
  } catch (e) {
    toast(`Could not load the report: ${e.message}`);
  }
}

function resetResults() {
  $("#results").hidden = true;
  $("#tab-processed").disabled = true;
  $("#tab-compare").disabled = true;
  $("#v-processed").removeAttribute("src");
  cP.removeAttribute("src");
  if ($("#stage").dataset.mode !== "original") setTab("original");
}

// ----------------------------------------------------------------- results
function renderResults(r, job) {
  $("#results").hidden = false;
  const s = r.summary, v = r.video;

  const dl = [
    ["annotated.mp4", "Annotated video"], ["report.json", "Report JSON"], ["tracks.csv", "Tracks CSV"],
    ["heatmap.png", "Heat map"], ["trajectories.png", "Trajectories"],
  ].filter(([n]) => job.files[n]);
  $("#downloads").innerHTML = dl.map(([n, l]) => `<a class="btn btn-ghost btn-sm" href="${esc(job.files[n])}?download=true" download><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v8m0 0 3.5-3.5M8 11 4.5 7.5M3 13h10"/></svg>${esc(l)}</a>`).join("");

  const tiles = [
    { label: "Video duration", kind: "meta", value: fmt(v.duration_s, 2), unit: "s", sub: `${v.width}×${v.height} · ${esc(v.codec)}` },
    { label: "Frame rate", kind: "meta", value: fmt(v.fps_used, 2), unit: "fps", sub: esc(v.fps_source) },
    { label: "Frames processed", kind: "measured", value: fmtInt(s.frames_processed), unit: "", sub: `at ${r.processing.width}×${r.processing.height} px` },
    { label: "Processing time", kind: "measured", value: fmt(s.processing_time_s, 1), unit: "s", sub: `${fmt(s.throughput_fps)} frames/s · ${esc(r.processing.detector)} detector` },
    { label: "Feature points tracked", kind: "measured", value: fmtInt(s.unique_features_tracked), unit: "", sub: `${fmt(s.mean_active_features, 0)} active per frame on average` },
    { label: r.processing.detector === "yolox" ? "Vehicles tracked" : "Moving objects tracked", kind: "measured", value: fmtInt(s.objects_tracked), unit: "", sub: `peak ${s.peak_concurrent_objects} at once` },
    { label: "Dominant direction", kind: "measured", value: s.dominant_direction ? `${s.dominant_direction.arrow} ${fmt(s.dominant_direction.angle_deg, 0)}°` : "–", unit: "", sub: `mean flow ${fmt(s.mean_flow_px_s)} px/s · ${s.dominant_direction ? esc(s.dominant_direction.label) : "no motion"}` },
    s.speed.calibrated && s.speed.median_kmh != null
      ? { label: "Median vehicle speed", kind: "estimate", value: `≈${fmt(s.speed.median_kmh, 0)}`, unit: "km/h", sub: `range ${fmt(s.speed.range_kmh[0], 0)}–${fmt(s.speed.range_kmh[1], 0)} km/h · ${s.speed.objects_with_estimate} vehicles` }
      : { label: "Speed", kind: "estimate", value: s.speed.calibrated ? "n/a" : "Not calibrated", unit: "", sub: s.speed.calibrated ? "no track stayed in the calibrated zone long enough" : "add a ground-plane calibration to estimate km/h" },
  ];
  const tagFor = { meta: '<span class="tag tag-meta">Metadata</span>', measured: '<span class="tag tag-measured">Measured</span>', estimate: '<span class="tag tag-estimate">Estimate</span>' };
  $("#stat-grid").innerHTML = tiles.map((t) => `
    <div class="stat">
      <div class="stat-top"><span class="stat-label">${t.label}</span>${tagFor[t.kind]}</div>
      <div class="stat-value">${t.value}${t.unit ? `<small>${t.unit}</small>` : ""}</div>
      <div class="stat-sub">${t.sub}</div>
    </div>`).join("");

  motionChart($("#chart-motion"), r.timeseries);
  roseChart($("#chart-rose"), r.direction_histogram.weights);
  trajectoryChart($("#chart-traj"), r, job);
  $("#heatmap-img").src = job.files["heatmap.png"] + `?t=${job.finished_at}`;
  tracksTable(r);
  $("#notes").innerHTML = r.notes.map((n) => `<li>${esc(n)}</li>`).join("");
}

// ---- motion over time (dual series, shared x)
function motionChart(el, ts) {
  const W = 900, H = 220, m = { l: 46, r: 40, t: 12, b: 26 };
  const n = ts.t.length;
  if (n < 2) { el.innerHTML = `<p class="muted">Not enough frames to chart.</p>`; return; }
  const tMax = ts.t[n - 1] || 1;
  const fMax = Math.max(1, ...ts.mean_flow_px_s) * 1.1;
  const oMax = Math.max(1, ...ts.active_objects);
  const x = (t) => m.l + (t / tMax) * (W - m.l - m.r);
  const yF = (v) => H - m.b - (v / fMax) * (H - m.t - m.b);
  const yO = (v) => H - m.b - (v / oMax) * (H - m.t - m.b);
  const line = ts.t.map((t, i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${yF(ts.mean_flow_px_s[i]).toFixed(1)}`).join("");
  const area = `${line}L${x(tMax)},${H - m.b}L${x(0)},${H - m.b}Z`;
  let step = "";
  ts.t.forEach((t, i) => { const yy = yO(ts.active_objects[i]).toFixed(1); step += i ? `H${x(t).toFixed(1)}V${yy}` : `M${x(t).toFixed(1)},${yy}`; });
  const ticks = niceTicks(fMax, 4);
  const xt = niceTicks(tMax, 6);
  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean flow in pixels per second and active vehicle count over time">
      <g class="grid">${ticks.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${yF(v)}" y2="${yF(v)}"/>`).join("")}</g>
      <g class="axis">
        ${ticks.map((v) => `<text x="${m.l - 8}" y="${yF(v) + 3}" text-anchor="end">${v}</text>`).join("")}
        ${xt.map((v) => `<text x="${x(v)}" y="${H - 8}" text-anchor="middle">${v}s</text>`).join("")}
        <text x="${W - m.r + 8}" y="${yO(oMax) + 3}">${oMax}</text><text x="${W - m.r + 8}" y="${yO(0) + 3}">0</text>
      </g>
      <path d="${area}" fill="rgba(255,178,40,.12)"/>
      <path d="${line}" fill="none" stroke="#ffb228" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="${step}" fill="none" stroke="#40d6e6" stroke-width="1.6" opacity=".9"/>
      <line class="cross" y1="${m.t}" y2="${H - m.b}" stroke="#ece8e1" stroke-opacity=".4" visibility="hidden"/>
      <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>
    </svg>
    <div class="tip" hidden></div>
    <div class="chart-legend"><span><i style="background:#ffb228"></i>Mean flow (px/s, left axis)</span><span><i style="background:#40d6e6"></i>Active vehicles (right axis)</span></div>`;
  const svg = $("svg", el), tip = $(".tip", el), cross = $(".cross", el);
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const sx = ((e.clientX - r.left) / r.width) * W;
    const t = Math.max(0, Math.min(tMax, ((sx - m.l) / (W - m.l - m.r)) * tMax));
    let i = 0; while (i < n - 1 && ts.t[i] < t) i++;
    cross.setAttribute("x1", x(ts.t[i])); cross.setAttribute("x2", x(ts.t[i])); cross.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.style.left = `${(x(ts.t[i]) / W) * r.width}px`;
    tip.style.top = `${(yF(ts.mean_flow_px_s[i]) / H) * r.height}px`;
    tip.innerHTML = `t ${ts.t[i].toFixed(2)} s<br>flow ${fmt(ts.mean_flow_px_s[i])} px/s<br>vehicles ${ts.active_objects[i]} · features ${ts.active_features[i]}<br>moving px ${fmt(ts.saliency_coverage[i], 2)}%`;
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}
function niceTicks(max, count) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) || mag * 10;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

// ---- direction rose
function roseChart(el, weights) {
  const S = 260, c = S / 2, R = 104;
  const max = Math.max(...weights, 1e-9);
  const wedges = weights.map((w, i) => {
    const a0 = (i * 22.5) * Math.PI / 180, a1 = ((i + 1) * 22.5) * Math.PI / 180;
    const r = Math.sqrt(w / max) * R;   // area-proportional
    if (r < 0.5) return "";
    const p = (a, rr) => `${(c + rr * Math.cos(a)).toFixed(1)},${(c - rr * Math.sin(a)).toFixed(1)}`;
    return `<path d="M${c},${c}L${p(a0, r)}A${r},${r} 0 0 0 ${p(a1, r)}Z" fill="${dirColor(i * 22.5 + 11.25)}" fill-opacity=".85" stroke="#16181d" stroke-width="1"><title>${(i * 22.5).toFixed(1)}–${((i + 1) * 22.5).toFixed(1)}°: ${(w * 100).toFixed(1)}%</title></path>`;
  }).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Direction rose of optical flow">
      ${[0.33, 0.66, 1].map((k) => `<circle cx="${c}" cy="${c}" r="${R * k}" fill="none" stroke="#262a31"/>`).join("")}
      <line x1="${c - R}" x2="${c + R}" y1="${c}" y2="${c}" stroke="#262a31"/><line y1="${c - R}" y2="${c + R}" x1="${c}" x2="${c}" stroke="#262a31"/>
      ${wedges}
      <g class="axis"><text x="${c}" y="${c - R - 8}" text-anchor="middle">up</text><text x="${c}" y="${c + R + 16}" text-anchor="middle">down</text>
      <text x="${c + R + 6}" y="${c + 3}">right</text><text x="${c - R - 6}" y="${c + 3}" text-anchor="end">left</text></g>
    </svg>`;
}

// ---- trajectories over the first frame
function trajectoryChart(el, r, job) {
  const W = r.processing.width, H = r.processing.height;
  const paths = r.tracks.map((t) => {
    const pts = t.trajectory_norm.map(([x, y]) => `${(x * W).toFixed(1)},${(y * H).toFixed(1)}`);
    if (pts.length < 2) return "";
    const end = t.trajectory_norm[t.trajectory_norm.length - 1];
    return `<path data-id="${t.id}" d="M${pts.join("L")}" fill="none" stroke="${dirColor(t.direction.angle_deg)}" stroke-width="2.4" stroke-linecap="round" opacity=".9"><title>#${t.id} ${esc(t.class)} · ${t.direction.arrow} ${t.direction.label}</title></path>
      <circle cx="${(end[0] * W).toFixed(1)}" cy="${(end[1] * H).toFixed(1)}" r="4" fill="${dirColor(t.direction.angle_deg)}"/>`;
  }).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Vehicle trajectories drawn over the first frame" style="border-radius:8px">
      <image href="${esc(job.files["thumbnail.jpg"])}?t=${job.finished_at}" width="${W}" height="${H}" opacity=".45"/>
      ${paths}
    </svg>`;
}

// ---- tracks table
function tracksTable(r) {
  const calibrated = r.summary.speed.calibrated;
  const tr = r.tracks;
  $("#tracks-sub").textContent = `${tr.length} confirmed track${tr.length === 1 ? "" : "s"}. Pixel values are measured in source-video pixels. `
    + (calibrated ? "Speeds are estimates: median and interquartile range of a sliding-window fit." : "No calibration was supplied, so no km/h is shown.");
  $("#tracks-table thead").innerHTML = `<tr><th>ID</th><th>Class</th><th class="num">Start</th><th class="num">Duration</th><th class="num">Net disp.</th><th>Direction</th><th class="num">LK flow</th><th class="num">Speed (est.)</th></tr>`;
  $("#tracks-table tbody").innerHTML = tr.length ? tr.map((t) => {
    let speed;
    if (!calibrated) speed = `<span class="na">needs calibration</span>`;
    else if (t.speed?.kind === "estimate") speed = `≈${fmt(t.speed.median_kmh, 0)} km/h <span class="na">(${fmt(t.speed.iqr_kmh[0], 0)}–${fmt(t.speed.iqr_kmh[1], 0)})</span>`;
    else speed = `<span class="na">outside zone</span>`;
    return `<tr data-id="${t.id}">
      <td>#${t.id}</td><td>${esc(t.class)}</td>
      <td class="num">${fmt(t.start_s, 2)} s</td><td class="num">${fmt(t.duration_s, 2)} s</td>
      <td class="num">${fmt(t.net_displacement_px, 0)} px</td>
      <td><span style="color:${dirColor(t.direction.angle_deg)}">${t.direction.arrow}</span> ${esc(t.direction.label)} <span class="na">${fmt(t.direction.angle_deg, 0)}°</span></td>
      <td class="num">${t.lk_velocity_px_s != null ? `${fmt(t.lk_velocity_px_s, 0)} px/s` : '<span class="na">–</span>'}</td>
      <td class="num">${speed}</td></tr>`;
  }).join("") : `<tr><td colspan="8"><span class="na">No moving objects were confirmed in this clip.</span></td></tr>`;
  const svg = $("#chart-traj svg");
  $$("#tracks-table tbody tr[data-id]").forEach((row) => {
    row.addEventListener("mouseenter", () => { svg?.classList.add("dim"); $(`path[data-id="${row.dataset.id}"]`, svg)?.classList.add("hl"); });
    row.addEventListener("mouseleave", () => { svg?.classList.remove("dim"); $$("path.hl", svg).forEach((p) => p.classList.remove("hl")); });
  });
}

// --------------------------------------------------------------------- boot
loadHealth();
loadDemos();
route();
