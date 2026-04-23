// ── State ──────────────────────────────────────────
let monitoring = true;
let wasAlerting = false;
let alarmTimer = null;

// ── Audio ─────────────────────────────────────────
// We keep ONE AudioContext alive for the whole session.
// It is created lazily on the first user gesture so the
// browser allows it.
let AC = null;

function ensureAudio() {
  if (!AC) AC = new (window.AudioContext || window.webkitAudioContext)();
  if (AC.state === "suspended") AC.resume();
  return AC;
}

function beep() {
  if (!document.getElementById("soundToggle").checked) return;
  try {
    const ac = ensureAudio();
    // Urgent alarm: 3 quick pulses at 900 Hz
    [0, 0.25, 0.5].forEach((delay) => {
      const osc = ac.createOscillator();
      const gain = ac.createGain();
      osc.connect(gain);
      gain.connect(ac.destination);
      osc.type = "square"; // harsher = more audible
      osc.frequency.value = 900;
      const t = ac.currentTime + delay;
      gain.gain.setValueAtTime(0.001, t);
      gain.gain.exponentialRampToValueAtTime(0.8, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
      osc.start(t);
      osc.stop(t + 0.2);
    });
  } catch (e) {
    console.error("beep error", e);
  }
}

function startAlarm() {
  if (alarmTimer) return;
  beep();
  alarmTimer = setInterval(beep, 3000);
}

function stopAlarm() {
  clearInterval(alarmTimer);
  alarmTimer = null;
}

// Test button — gives a guaranteed user-gesture unlock + audible confirmation
document.getElementById("testSoundBtn").addEventListener("click", () => {
  try {
    ensureAudio();
    beep();
    addLog("🔊 Sound test fired", false);
  } catch (e) {
    addLog("❌ Audio failed: " + e.message);
  }
});

// ── Alert log ─────────────────────────────────────
function addLog(msg, isAlert = true) {
  const box = document.getElementById("logBox");
  const t = new Date().toTimeString().slice(0, 8);
  const el = document.createElement("div");
  el.className = "log-entry";
  el.innerHTML = `<span class="log-time">${t}</span><span class="log-msg ${isAlert ? "" : "ok"}">${msg}</span>`;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  while (box.children.length > 60) box.removeChild(box.firstChild);
}

// ── Clock ─────────────────────────────────────────
setInterval(() => {
  document.getElementById("clock").textContent = new Date()
    .toTimeString()
    .slice(0, 8);
}, 1000);

// ── Start / Stop ──────────────────────────────────
document.getElementById("toggleBtn").addEventListener("click", () => {
  monitoring = !monitoring;
  const btn = document.getElementById("toggleBtn");
  const paused = document.getElementById("pausedOverlay");
  const abox = document.getElementById("alertBox");
  const badge = document.getElementById("statusBadge");

  if (!monitoring) {
    btn.textContent = "▶ START";
    btn.classList.remove("stopping");
    paused.classList.add("visible");
    abox.className = "alert-box paused-state";
    abox.textContent = "⏸  MONITORING PAUSED";
    badge.className = "status-badge paused";
    badge.textContent = "⏸  PAUSED";
    stopAlarm();
    wasAlerting = false;
    addLog("Monitoring paused", false);
  } else {
    btn.textContent = "■ STOP";
    btn.classList.add("stopping");
    paused.classList.remove("visible");
    abox.className = "alert-box";
    abox.textContent = "⬤  SYSTEM NOMINAL";
    badge.className = "status-badge";
    badge.textContent = "INITIALISING";
    addLog("Monitoring resumed", false);
  }
});

// ── Video frame polling ───────────────────────────
const canvas = document.getElementById("feed");
const ctx2d = canvas.getContext("2d");
const img = new Image();

img.onload = () => {
  canvas.width = img.naturalWidth || canvas.offsetWidth;
  canvas.height = img.naturalHeight || canvas.offsetHeight;
  if (monitoring) ctx2d.drawImage(img, 0, 0, canvas.width, canvas.height);
};

(async function pollFrame() {
  if (monitoring) {
    try {
      const r = await fetch("/frame?t=" + Date.now());
      const d = await r.json();
      if (d.image) img.src = "data:image/jpeg;base64," + d.image;
    } catch (e) {}
  }
  setTimeout(pollFrame, 40);
})();

// ── Gauge ─────────────────────────────────────────
function setGauge(p) {
  const f = document.getElementById("gauge-fill");
  const t = document.getElementById("gaugeText");
  f.setAttribute("stroke-dashoffset", (283 - p * 283).toFixed(1));
  const col = `rgb(${Math.round(p * 255)},${Math.round((1 - p) * 200)},80)`;
  f.setAttribute("stroke", col);
  t.textContent = p.toFixed(2);
  t.setAttribute("fill", col);
}

// ── Stats polling ─────────────────────────────────
(async function pollStats() {
  if (!monitoring) {
    setTimeout(pollStats, 300);
    return;
  }

  try {
    const r = await fetch("/stats");
    const data = await r.json();

    // Face pill
    const pill = document.getElementById("facePill");
    if (data.face_detected) {
      pill.classList.add("active");
      document.getElementById("faceLabel").textContent = "Face locked";
    } else {
      pill.classList.remove("active");
      document.getElementById("faceLabel").textContent = "No face";
    }

    const abox = document.getElementById("alertBox");
    const aover = document.getElementById("alertOverlay");
    const badge = document.getElementById("statusBadge");

    if (data.alert) {
      abox.className = "alert-box triggered";
      abox.textContent = "⚠  DROWSINESS DETECTED";
      aover.classList.add("active");
      badge.className = "status-badge drowsy";
      badge.textContent = "⬤  DROWSY";
      if (!wasAlerting) {
        addLog("⚠ Drowsiness detected!");
        startAlarm();
      }
      wasAlerting = true;
    } else {
      abox.className = "alert-box";
      abox.textContent = "⬤  SYSTEM NOMINAL";
      aover.classList.remove("active");
      badge.className = data.face_detected
        ? "status-badge safe"
        : "status-badge";
      badge.textContent = data.face_detected ? "⬤  ALERT" : "SEARCHING…";
      if (wasAlerting) {
        addLog("✓ Alert cleared — driver alert", false);
        stopAlarm();
      }
      wasAlerting = false;
    }

    const rawEl = document.getElementById("rawLabel");
    rawEl.textContent = data.raw_label;
    rawEl.className =
      "card-value " +
      (data.raw_label === "drowsy"
        ? "danger"
        : data.raw_label === "notdrowsy"
          ? "safe"
          : "");

    document.getElementById("awakeVal").textContent =
      data.notdrowsy_prob.toFixed(2);
    document.getElementById("drowsyVal").textContent =
      data.drowsy_prob.toFixed(2);
    document.getElementById("awakeBar").style.width =
      (data.notdrowsy_prob * 100).toFixed(1) + "%";
    document.getElementById("drowsyBar").style.width =
      (data.drowsy_prob * 100).toFixed(1) + "%";
    setGauge(data.smooth_drowsy_prob);
  } catch (e) {
    console.warn(e);
  }

  setTimeout(pollStats, 300);
})();
