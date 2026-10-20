"use strict";

// ------------------------------------------------------------------ helpers

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const main = $("#main");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 3500);
}

function clock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function duration(sec) {
  sec = Math.round(sec || 0);
  if (sec < 60) return `${sec}s`;
  const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m} min`;
}

function dateLabel(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return sameDay ? `Today ${time}` : `${d.toLocaleDateString([], { day: "numeric", month: "short", year: d.getFullYear() === today.getFullYear() ? undefined : "numeric" })} ${time}`;
}

const STATUS_LABEL = { recording: "Recording", queued: "Queued", processing: "Processing", done: "", error: "Error" };

function badge(status) {
  if (!STATUS_LABEL[status]) return "";
  const pulse = status === "recording" || status === "processing" ? '<span class="pulse"></span>' : "";
  return `<span class="badge ${status}">${pulse}${STATUS_LABEL[status]}</span>`;
}

function speakerColor(label, order) {
  const i = order.indexOf(label);
  return `var(--s${(i < 0 ? 0 : i) % 8 + 1})`;
}

// Minimal Markdown for summaries (headings, bullets, tasks, bold, code). Input is escaped first.
function markdown(md) {
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`(.+?)`/g, "<code>$1</code>");
  const out = [];
  let inList = false;
  for (const raw of md.split("\n")) {
    const line = raw.trimEnd();
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { out.push("<ul>"); inList = true; }
      const task = li[1].match(/^\[( |x|X)\]\s*(.*)$/);
      if (task) out.push(`<li class="task${task[1] === " " ? "" : " done"}">${inline(task[2])}</li>`);
      else out.push(`<li>${inline(li[1])}</li>`);
      continue;
    }
    if (inList) { out.push("</ul>"); inList = false; }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) out.push(`<h2>${inline(h[2])}</h2>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (inList) out.push("</ul>");
  return out.join("\n");
}

// ------------------------------------------------------------------ state

const state = {
  status: null,
  meetings: [],
  route: { name: "home" },
  meeting: null, // data for the open meeting page
};

// ------------------------------------------------------------------ sidebar

let searchTimer;
$("#search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(refreshMeetings, 250);
});

async function refreshMeetings() {
  try {
    state.meetings = await api("GET", `/api/meetings?q=${encodeURIComponent($("#search").value)}`);
  } catch (e) {
    return;
  }
  renderSidebar();
}

function renderSidebar() {
  const activeId = state.route.id;
  const list = $("#meetingList");
  if (!state.meetings.length) {
    list.innerHTML = `<div class="empty" style="padding:24px 8px">${$("#search").value ? "No matches" : "No meetings yet"}</div>`;
    return;
  }
  list.innerHTML = state.meetings.map((m) => `
    <a class="meeting-item ${m.id === activeId ? "active" : ""}" href="#/${m.status === "recording" ? "live" : "m"}/${m.id}">
      <div class="t">${esc(m.title)}</div>
      <div class="m">${dateLabel(m.created_at)}${m.duration ? ` · ${duration(m.duration)}` : ""} ${badge(m.status)}</div>
    </a>`).join("");
  $("#settingsLink").classList.toggle("active", state.route.name === "settings");
}

async function refreshStatus() {
  try {
    state.status = await api("GET", "/api/status");
  } catch {
    $("#sysStatus").textContent = "Server unreachable";
    return;
  }
  const s = state.status;
  const parts = [];
  parts.push(s.gpus ? `${s.gpus} GPU${s.gpus > 1 ? "s" : ""}` : "CPU only");
  if (s.whisper.running) parts.push("model loaded");
  if (s.processing) parts.push(`processing${s.queued ? ` (+${s.queued})` : ""}`);
  if (s.autodetect !== "off") parts.push(`auto-detect: ${s.autodetect}`);
  $("#sysStatus").textContent = parts.join(" · ");
}

// ------------------------------------------------------------------ websocket

let ws, wsDelay = 1000;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { wsDelay = 1000; };
  ws.onclose = () => { setTimeout(connect, wsDelay); wsDelay = Math.min(wsDelay * 2, 15000); };
  ws.onmessage = (ev) => onEvent(JSON.parse(ev.data));
}

function onEvent(e) {
  const r = state.route;
  switch (e.type) {
    case "levels":
      if (r.name === "live" && r.id === e.meeting) updateLive(e);
      break;
    case "caption":
      if (r.name === "live" && r.id === e.meeting) addCaption(e, true);
      break;
    case "status":
      refreshMeetings();
      refreshStatus();
      if (r.id === e.meeting) {
        if (r.name === "live" && e.status !== "recording") location.hash = `#/m/${e.meeting}`;
        else if (r.name === "m") loadMeeting(e.meeting, true);
      }
      if (e.status === "done") toast("Transcript ready");
      if (e.status === "error") toast(`Processing failed: ${e.error || ""}`, "error");
      if (e.status === "recording" && r.name === "home") location.hash = `#/live/${e.meeting}`;
      break;
    case "progress":
      if (r.name === "m" && r.id === e.meeting) updateProgress(e.progress, e.stage);
      break;
    case "summary":
      if (r.name === "m" && r.id === e.meeting) loadMeeting(e.meeting, true);
      if (e.status === "error") toast(`Summary failed: ${e.error}`, "error");
      break;
    case "error":
      toast(e.message, "error");
      break;
  }
}

// ------------------------------------------------------------------ router

function parseRoute() {
  const h = location.hash.replace(/^#\/?/, "");
  const [name, id] = h.split("/");
  if (name === "live" && id) return { name: "live", id };
  if (name === "m" && id) return { name: "m", id };
  if (name === "settings") return { name: "settings" };
  return { name: "home" };
}

async function route() {
  state.route = parseRoute();
  renderSidebar();
  main.scrollTop = 0;
  const r = state.route;
  try {
    if (r.name === "home") await renderHome();
    else if (r.name === "live") await renderLive(r.id);
    else if (r.name === "m") await loadMeeting(r.id);
    else if (r.name === "settings") await renderSettings();
  } catch (e) {
    main.innerHTML = `<div class="page"><div class="alert error">${esc(e.message)}</div></div>`;
  }
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ home

async function renderHome() {
  await refreshStatus();
  const s = state.status;
  if (s.recording) { location.hash = `#/live/${s.recording.id}`; return; }
  const [devices, settings] = await Promise.all([api("GET", "/api/devices"), api("GET", "/api/settings")]);

  const opt = (items, selected, defLabel) =>
    `<option value="">${esc(defLabel)}</option>` +
    items.map((d) => `<option value="${esc(d.name)}" ${d.name === selected ? "selected" : ""}>${esc(d.label)}</option>`).join("");
  const defMic = devices.mics.find((d) => d.name === devices.default_mic);
  const defMon = devices.monitors.find((d) => d.name === devices.default_monitor);

  const checks = [
    [s.capture_available, "Audio capture (PulseAudio / PipeWire)", "Not available here. You can still upload recordings."],
    [s.whisper_ready, "whisper.cpp and model installed", "Run <code>./install.sh</code>"],
    [s.hf_token, "Hugging Face token for speaker identification", 'Add it in <a href="#/settings"><u>Settings</u></a>'],
    [s.summary_provider !== "none", "AI summary provider", 'Optional: choose Claude or Ollama in <a href="#/settings"><u>Settings</u></a>'],
  ];
  const allGood = checks.slice(0, 3).every((c) => c[0]);

  main.innerHTML = `
  <div class="page">
    <h1 style="margin-bottom:6px">New meeting</h1>
    <p class="muted" style="margin:0 0 22px">Records your microphone and the call audio playing on this computer. Live captions while you talk; a full speaker-labelled transcript when you stop.</p>
    <form class="card" id="startForm">
      <label class="field"><span>Title</span>
        <input type="text" name="title" placeholder="e.g. Weekly sync with client" autocomplete="off"></label>
      <div class="grid2">
        <label class="field"><span>Your microphone</span>
          <select name="mic_source" ${devices.available ? "" : "disabled"}>${opt(devices.mics, settings.mic_source, `System default${defMic ? ` (${defMic.label})` : ""}`)}</select>
          <label class="check"><input type="checkbox" name="no_mic"> <small>Don't record my microphone</small></label>
        </label>
        <label class="field"><span>Call audio (what you hear)</span>
          <select name="call_source" ${devices.available ? "" : "disabled"}>${opt(devices.monitors, settings.call_source, `Default output${defMon ? ` (${defMon.label.replace(/^Monitor of /, "")})` : ""}`)}</select>
          <small>Remote participants are captured from here and separated by voice.</small>
        </label>
      </div>
      <div class="row" style="margin-top:6px">
        <button class="btn btn-rec" type="submit" ${s.capture_available ? "" : "disabled"}><span class="brand-dot" style="background:#fff"></span> Start recording</button>
        <span class="muted">Recording as <strong>${esc(settings.user_name)}</strong> · <a href="#/settings"><u>change</u></a></span>
      </div>
    </form>
    <div class="card">
      <h2>${allGood ? "Ready" : "Setup"}</h2>
      <ul class="checklist">${checks.map(([ok, label, fix]) => `
        <li><span class="ic ${ok ? "ok" : "bad"}">${ok ? "✓" : "!"}</span>
          <span>${label}${ok ? "" : ` <span class="muted">· ${fix}</span>`}</span></li>`).join("")}
      </ul>
      <p class="faint" style="margin:14px 0 0"><small>Whisper on ${s.whisper.gpu === null ? "CPU" : `GPU ${s.whisper.gpu}`} · speaker identification on ${esc(s.diarization_device)} · auto-detect calls: ${s.autodetect}</small></p>
    </div>
  </div>`;

  $("#startForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = new FormData(ev.target);
    const btn = ev.target.querySelector("button[type=submit]");
    btn.disabled = true;
    try {
      const meta = await api("POST", "/api/recordings", {
        title: f.get("title"),
        mic_source: f.get("mic_source") ?? "",
        call_source: f.get("call_source") ?? "",
        use_mic: !f.get("no_mic"),
      });
      location.hash = `#/live/${meta.id}`;
      refreshMeetings();
    } catch (e) {
      toast(e.message, "error");
      btn.disabled = false;
    }
  });
}

// ------------------------------------------------------------------ live

async function renderLive(id) {
  const data = await api("GET", `/api/meetings/${id}`);
  await refreshStatus();
  const rec = state.status.recording;
  if (!rec || rec.id !== id) { location.hash = `#/m/${id}`; return; }

  const streamLabel = { mic: "Your microphone", call: "Call audio" };
  main.innerHTML = `
  <div class="page">
    <div class="live-head">
      <span class="rec-dot"></span>
      <span class="timer" id="timer">${clock(rec.elapsed)}</span>
      <div style="flex:1;min-width:0"><h1 style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(data.meta.title)}</h1></div>
      <button class="btn btn-rec" id="stopBtn">■ Stop</button>
    </div>
    <div class="meters">${rec.streams.map((st) => `
      <div class="meter" id="meter-${st}">
        <div class="lbl"><span>${streamLabel[st]}</span><span class="err-msg"></span></div>
        <div class="track"><div class="fill"></div></div>
      </div>`).join("")}
    </div>
    <div class="row" style="margin-bottom:12px"><h2>Live captions</h2><span class="spacer"></span><small class="faint" id="backlog"></small></div>
    <div class="captions" id="captions"><div class="empty" id="capEmpty">Listening… captions appear a few seconds after someone speaks.</div></div>
  </div>`;
  for (const c of data.live) addCaption(c, false);
  $("#stopBtn").addEventListener("click", async () => {
    $("#stopBtn").disabled = true;
    $("#stopBtn").textContent = "Stopping…";
    try {
      await api("POST", "/api/recordings/stop");
      location.hash = `#/m/${id}`;
    } catch (e) {
      toast(e.message, "error");
      $("#stopBtn").disabled = false;
    }
  });
}

function updateLive(e) {
  const t = $("#timer");
  if (t) t.textContent = clock(e.elapsed);
  for (const [st, lvl] of Object.entries(e.levels)) {
    const m = $(`#meter-${st}`);
    if (!m) continue;
    // perceptual scale: speech RMS is usually 0.01-0.2
    const pct = Math.min(100, Math.max(0, (Math.log10(lvl + 1e-4) + 3.3) / 2.6 * 100));
    $(".fill", m).style.width = `${pct}%`;
    const err = e.errors?.[st];
    m.classList.toggle("err", !!err);
    $(".err-msg", m).textContent = err ? "capture error" : "";
  }
  const b = $("#backlog");
  if (b) b.textContent = e.backlog > 1 ? `${e.backlog} chunks waiting` : "";
}

function addCaption(c, scroll) {
  const box = $("#captions");
  if (!box) return;
  $("#capEmpty")?.remove();
  const color = c.stream === "mic" ? "var(--s1)" : "var(--s2)";
  const el = document.createElement("div");
  el.className = "caption";
  el.innerHTML = `<div class="when">${clock(c.start)}</div>
    <div><div class="who" style="color:${color}">${esc(c.stream === "mic" ? "You" : "Call")}</div><div>${esc(c.text)}</div></div>`;
  box.appendChild(el);
  if (scroll) {
    const nearBottom = main.scrollHeight - main.scrollTop - main.clientHeight < 200;
    if (nearBottom) main.scrollTop = main.scrollHeight;
  }
}

// ------------------------------------------------------------------ meeting

async function loadMeeting(id, keepTab = false) {
  const data = await api("GET", `/api/meetings/${id}`);
  if (data.meta.status === "recording" && state.status?.recording?.id === id) {
    location.hash = `#/live/${id}`;
    return;
  }
  const prevTab = keepTab && state.meeting?.meta.id === id ? state.meeting.tab : null;
  const audio = $("#audio");
  const resume = keepTab && audio ? { t: audio.currentTime, playing: !audio.paused } : null;
  state.meeting = { ...data, tab: prevTab || (data.transcript ? "transcript" : "live") };
  renderMeeting();
  if (resume && $("#audio")) {
    $("#audio").currentTime = resume.t;
    if (resume.playing) $("#audio").play();
  }
}

function names() {
  return state.meeting.meta.speakers || {};
}
function nameOf(label) {
  return names()[label] || label;
}

function speakerOrder() {
  const segs = state.meeting.transcript?.segments || [];
  const seen = [];
  for (const s of segs) if (!seen.includes(s.speaker)) seen.push(s.speaker);
  return seen.sort((a, b) => (a === "Me" ? -1 : b === "Me" ? 1 : a.localeCompare(b, undefined, { numeric: true })));
}

function renderMeeting() {
  const { meta, transcript, live, summary, has_audio } = state.meeting;
  const busy = ["queued", "processing"].includes(meta.status);
  const tab = state.meeting.tab;
  const langNames = { id: "Indonesian", en: "English", ms: "Malay", jw: "Javanese", su: "Sundanese" };

  main.innerHTML = `
  <div class="page">
    <div class="m-head">
      <h1 class="m-title" id="mTitle" contenteditable="true" spellcheck="false">${esc(meta.title)}</h1>
      <div class="menu">
        <button class="btn" id="menuBtn" aria-haspopup="true">Export & more ▾</button>
        <div class="menu-list" id="menuList" hidden>
          ${transcript ? `
          <a href="/api/meetings/${meta.id}/export/docx">Word document (.docx)</a>
          <a href="/api/meetings/${meta.id}/export/md">Markdown (.md)</a>
          <a href="/api/meetings/${meta.id}/export/txt">Text (.txt)</a>
          <a href="/api/meetings/${meta.id}/export/srt">Subtitles (.srt)</a>
          ${has_audio ? `<a href="/api/meetings/${meta.id}/audio" download="${esc(meta.title)}.opus">Audio (.opus)</a>` : ""}
          <hr>` : ""}
          <button id="reprocessBtn" ${busy ? "disabled" : ""}>Re-run transcription</button>
          <button id="deleteBtn" class="btn-danger" ${busy ? "disabled" : ""}>Delete meeting</button>
        </div>
      </div>
    </div>
    <div class="m-sub">
      <span>${dateLabel(meta.created_at)}</span>
      ${meta.duration ? `<span>· ${duration(meta.duration)}</span>` : ""}
      ${meta.language ? `<span>· ${esc(langNames[meta.language] || meta.language)}</span>` : ""}
      ${meta.mode === "upload" ? "<span>· uploaded</span>" : ""}
      ${badge(meta.status)}
    </div>
    ${busy ? `<div class="progress card"><div class="row" style="margin-bottom:8px"><strong id="stage">${meta.status === "queued" ? "Waiting to process…" : esc(meta.stage || "Processing…")}</strong><span class="spacer"></span><span class="muted" id="pct">${Math.round((meta.progress || 0) * 100)}%</span></div><div class="bar"><span id="progBar" style="width:${(meta.progress || 0) * 100}%"></span></div><small class="faint">You can close this page; processing continues in the background.</small></div>` : ""}
    ${meta.status === "error" ? `<div class="alert error"><strong>Processing failed.</strong> ${esc(meta.error)} <button class="btn btn-sm" id="retryBtn">Try again</button></div>` : ""}
    ${meta.status === "done" && meta.error ? `<div class="alert">${esc(meta.error)}</div>` : ""}
    ${has_audio ? `<div class="player"><audio id="audio" controls preload="metadata" src="/api/meetings/${meta.id}/audio"></audio></div>` : ""}
    <div class="tabs">
      ${transcript ? `<button class="tab ${tab === "transcript" ? "active" : ""}" data-tab="transcript">Transcript</button>` : ""}
      <button class="tab ${tab === "summary" ? "active" : ""}" data-tab="summary">Summary</button>
      ${live.length ? `<button class="tab ${tab === "live" ? "active" : ""}" data-tab="live">Live captions</button>` : ""}
    </div>
    <div id="tabBody"></div>
  </div>`;

  // title edit
  const t = $("#mTitle");
  t.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); t.blur(); } });
  t.addEventListener("blur", async () => {
    const v = t.textContent.trim();
    if (v && v !== meta.title) {
      state.meeting.meta = await api("PATCH", `/api/meetings/${meta.id}`, { title: v });
      refreshMeetings();
    } else t.textContent = meta.title;
  });

  // menu
  $("#menuBtn").addEventListener("click", (e) => { e.stopPropagation(); $("#menuList").hidden = !$("#menuList").hidden; });
  $("#reprocessBtn").addEventListener("click", reprocess);
  $("#retryBtn")?.addEventListener("click", reprocess);
  $("#deleteBtn").addEventListener("click", async () => {
    if (!confirm(`Delete "${meta.title}" with its audio and transcript? This cannot be undone.`)) return;
    try {
      await api("DELETE", `/api/meetings/${meta.id}`);
      toast("Meeting deleted");
      location.hash = "#/";
      refreshMeetings();
    } catch (e) { toast(e.message, "error"); }
  });

  $$(".tab").forEach((b) => b.addEventListener("click", () => {
    state.meeting.tab = b.dataset.tab;
    $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
    renderTab();
  }));
  renderTab();

  const audio = $("#audio");
  if (audio) audio.addEventListener("timeupdate", highlightPlaying);
}

async function reprocess() {
  const meta = state.meeting.meta;
  if (state.meeting.transcript && !confirm("Re-run transcription and speaker identification? Text edits to the transcript will be replaced (speaker names are kept).")) return;
  try {
    await api("POST", `/api/meetings/${meta.id}/reprocess`);
    loadMeeting(meta.id, true);
  } catch (e) { toast(e.message, "error"); }
}

function updateProgress(p, stage) {
  const bar = $("#progBar");
  if (!bar) { loadMeeting(state.route.id, true); return; }
  bar.style.width = `${p * 100}%`;
  $("#pct").textContent = `${Math.round(p * 100)}%`;
  if (stage) $("#stage").textContent = stage.charAt(0).toUpperCase() + stage.slice(1) + "…";
}

function renderTab() {
  const body = $("#tabBody");
  const tab = state.meeting.tab;
  if (tab === "transcript") renderTranscript(body);
  else if (tab === "summary") renderSummary(body);
  else renderLiveCaptions(body);
}

function renderTranscript(body) {
  const { meta, transcript } = state.meeting;
  const segs = transcript.segments;
  const order = speakerOrder();
  const talk = {};
  for (const s of segs) talk[s.speaker] = (talk[s.speaker] || 0) + (s.end - s.start);
  const total = Object.values(talk).reduce((a, b) => a + b, 0) || 1;

  body.innerHTML = `
    <div class="speakers">${order.map((sp) => `
      <label class="speaker-chip" title="Rename ${esc(sp)}">
        <span class="dot" style="background:${speakerColor(sp, order)}"></span>
        <input value="${esc(nameOf(sp))}" data-speaker="${esc(sp)}" aria-label="Name for ${esc(sp)}">
        <span class="stat">${Math.round((talk[sp] / total) * 100)}%</span>
      </label>`).join("")}
    </div>
    ${segs.length ? "" : '<div class="empty">No speech was found in this recording.</div>'}
    <div class="transcript" id="transcript">${segs.map((s, i) => `
      <div class="para" data-i="${i}" data-start="${s.start}" data-end="${s.end}">
        <div class="when" title="Play from here">${clock(s.start)}</div>
        <div>
          <div class="who" style="color:${speakerColor(s.speaker, order)}" title="Change speaker">${esc(nameOf(s.speaker))}</div>
          <div class="txt" title="Double-click to edit">${esc(s.text)}</div>
        </div>
      </div>`).join("")}
    </div>
    <p class="faint" style="margin-top:24px"><small>Tip: rename speakers above · click a time to play · double-click text to correct it · click a name to reassign a paragraph.</small></p>`;

  $$(".speaker-chip input", body).forEach((inp) => {
    inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); });
    inp.addEventListener("change", async () => {
      const sp = inp.dataset.speaker;
      try {
        state.meeting.meta = await api("PATCH", `/api/meetings/${meta.id}`, { speakers: { [sp]: inp.value } });
        $$(".para", body).forEach((p) => {
          const seg = segs[p.dataset.i];
          if (seg.speaker === sp) $(".who", p).textContent = nameOf(sp);
        });
      } catch (e) { toast(e.message, "error"); }
    });
  });

  const tr = $("#transcript");
  tr.addEventListener("click", (e) => {
    const para = e.target.closest(".para");
    if (!para) return;
    if (e.target.classList.contains("when")) {
      const a = $("#audio");
      if (a) { a.currentTime = Math.max(0, parseFloat(para.dataset.start) - 0.3); a.play(); }
    } else if (e.target.classList.contains("who")) {
      pickSpeaker(e.target, para, order);
    }
  });
  tr.addEventListener("dblclick", (e) => {
    const txt = e.target.closest(".txt");
    if (!txt || txt.isContentEditable) return;
    const para = txt.closest(".para");
    const i = +para.dataset.i;
    txt.contentEditable = "true";
    txt.focus();
    const done = async (save) => {
      txt.contentEditable = "false";
      const v = txt.textContent.trim();
      if (save && v && v !== segs[i].text) {
        try {
          await api("PATCH", `/api/meetings/${meta.id}/segments`, { index: i, text: v });
          segs[i].text = v;
        } catch (err) { toast(err.message, "error"); txt.textContent = segs[i].text; }
      } else txt.textContent = segs[i].text;
    };
    txt.addEventListener("blur", () => done(true), { once: true });
    txt.addEventListener("keydown", function k(ev) {
      if (ev.key === "Escape") { txt.removeEventListener("keydown", k); txt.textContent = segs[i].text; txt.blur(); }
      if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); txt.removeEventListener("keydown", k); txt.blur(); }
    });
  });
}

function pickSpeaker(whoEl, para, order) {
  const { meta, transcript } = state.meeting;
  const i = +para.dataset.i;
  const seg = transcript.segments[i];
  const nextNum = Math.max(0, ...order.map((s) => +(s.match(/(\d+)$/)?.[1] || 0))) + 1;
  const sel = document.createElement("select");
  sel.innerHTML = order.map((sp) => `<option value="${esc(sp)}" ${sp === seg.speaker ? "selected" : ""}>${esc(nameOf(sp))}</option>`).join("")
    + `<option value="Speaker ${nextNum}">+ New speaker</option>`;
  sel.style.width = "auto";
  sel.style.height = "28px";
  whoEl.replaceWith(sel);
  sel.focus();
  const finish = async () => {
    const v = sel.value;
    if (v !== seg.speaker) {
      try {
        await api("PATCH", `/api/meetings/${meta.id}/segments`, { index: i, speaker: v });
        seg.speaker = v;
      } catch (e) { toast(e.message, "error"); }
    }
    renderTab();
  };
  sel.addEventListener("change", finish);
  sel.addEventListener("blur", () => setTimeout(() => sel.isConnected && renderTab(), 150));
}

let lastPlaying = null;
function highlightPlaying() {
  const t = $("#audio").currentTime;
  const paras = $$(".para");
  let cur = null;
  for (const p of paras) {
    if (t >= +p.dataset.start - 0.2 && t <= +p.dataset.end + 0.5) { cur = p; break; }
  }
  if (cur === lastPlaying) return;
  lastPlaying?.classList.remove("playing");
  cur?.classList.add("playing");
  lastPlaying = cur;
}

function renderSummary(body) {
  const { meta, summary, transcript } = state.meeting;
  const running = meta.summary_status === "running";
  const provider = state.status?.summary_provider || "none";
  if (!transcript) {
    body.innerHTML = '<div class="empty">The summary is available after the transcript is ready.</div>';
    return;
  }
  body.innerHTML = `
    ${meta.summary_error && !running ? `<div class="alert error">${esc(meta.summary_error)}</div>` : ""}
    ${provider === "none" ? `<div class="alert">Choose Claude or Ollama in <a href="#/settings"><u>Settings</u></a> to generate summaries.</div>` : ""}
    <div class="row" style="margin-bottom:16px">
      <button class="btn ${summary ? "" : "btn-primary"}" id="sumBtn" ${running || provider === "none" ? "disabled" : ""}>
        ${running ? "Summarizing…" : summary ? "Regenerate summary" : "Generate summary"}</button>
      ${running ? '<span class="pulse" style="color:var(--accent)"></span>' : ""}
      <span class="faint"><small>${provider === "claude" ? "Claude (Anthropic API): the transcript is sent to Anthropic." : provider === "ollama" ? "Ollama: runs locally, nothing leaves this computer." : ""}</small></span>
    </div>
    ${summary ? `<div class="card summary">${markdown(summary)}</div>` : running ? "" : '<div class="empty">No summary yet.</div>'}`;
  $("#sumBtn")?.addEventListener("click", async () => {
    $("#sumBtn").disabled = true;
    $("#sumBtn").textContent = "Summarizing…";
    try {
      await api("POST", `/api/meetings/${meta.id}/summary`);
    } catch (e) {
      toast(e.message, "error");
    }
    loadMeeting(meta.id, true);
  });
}

function renderLiveCaptions(body) {
  const { live } = state.meeting;
  body.innerHTML = `<p class="muted" style="margin-top:0">Rough captions made during the meeting. The transcript tab has the refined version with speaker names once processing finishes.</p>
    <div class="captions" id="captions"></div>`;
  for (const c of live) addCaption(c, false);
}

// ------------------------------------------------------------------ settings

async function renderSettings() {
  const [s, status, ollama] = await Promise.all([api("GET", "/api/settings"), api("GET", "/api/status"), api("GET", "/api/ollama/models")]);
  const models = status.models;
  const sel = (name, options, value) => `<select name="${name}">${options.map(([v, l]) => `<option value="${esc(v)}" ${String(v) === String(value) ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
  const gpuOpts = [["auto", "Automatic"], ["cpu", "CPU"], ...Array.from({ length: status.gpus }, (_, i) => [String(i), `GPU ${i}`])];
  const langs = [["auto", "Auto-detect (mixed Indonesian / English)"], ["id", "Indonesian"], ["en", "English"], ["ms", "Malay"], ["jw", "Javanese"], ["zh", "Chinese"], ["ja", "Japanese"]];

  main.innerHTML = `
  <div class="page">
    <h1 style="margin-bottom:20px">Settings</h1>
    <form id="settingsForm">
      <div class="card settings-section">
        <h2>You</h2>
        <label class="field"><span>Your name</span><input type="text" name="user_name" value="${esc(s.user_name)}">
          <small>Your microphone is labelled with this name in transcripts.</small></label>
      </div>

      <div class="card settings-section">
        <h2>Transcription</h2>
        <div class="grid2">
          <label class="field"><span>Language</span>${sel("language", langs, s.language)}</label>
          <label class="field"><span>Whisper model</span>${sel("whisper_model", models.map((m) => [m, m.replace(/^ggml-|\.bin$/g, "")]), s.whisper_model)}
            <small>Download more with <code>scripts/download_models.sh &lt;name&gt;</code></small></label>
          <label class="field"><span>Live caption model</span>${sel("live_model", [["", "Same as above"], ...models.map((m) => [m, m.replace(/^ggml-|\.bin$/g, "")])], s.live_model)}
            <small>A smaller model here keeps captions fast on CPU-only computers.</small></label>
          <label class="field"><span>Whisper runs on</span>${sel("whisper_device", gpuOpts, s.whisper_device)}</label>
          <label class="field"><span>CPU threads</span><input type="number" name="whisper_threads" min="1" max="64" value="${s.whisper_threads}"></label>
        </div>
      </div>

      <div class="card settings-section">
        <h2>Speaker identification (pyannote)</h2>
        <label class="field"><span>Hugging Face token</span>
          <input type="password" name="hf_token" value="${esc(s.hf_token)}" placeholder="${status.hf_token ? "Using token from environment / huggingface-cli login" : "hf_…"}" autocomplete="off">
          <small>Create one at huggingface.co/settings/tokens, then accept the terms on <a href="https://huggingface.co/${esc(s.diarization_model)}" target="_blank" rel="noopener"><u>huggingface.co/${esc(s.diarization_model)}</u></a>.</small></label>
        <div class="grid2">
          <label class="field"><span>Runs on</span>${sel("diarization_device", gpuOpts, s.diarization_device)}
            <small>Automatic: the GPU whisper doesn't use, when there are two.</small></label>
          <label class="field"><span>Model</span><input type="text" name="diarization_model" value="${esc(s.diarization_model)}"></label>
          <label class="field"><span>Min speakers on the call</span><input type="number" name="min_speakers" min="0" value="${s.min_speakers}"><small>0 = automatic</small></label>
          <label class="field"><span>Max speakers on the call</span><input type="number" name="max_speakers" min="0" value="${s.max_speakers}"><small>0 = automatic</small></label>
        </div>
        <label class="check"><input type="checkbox" name="echo_dedup" ${s.echo_dedup ? "checked" : ""}> Remove call audio that leaks into my microphone (when not using headphones)</label>
      </div>

      <div class="card settings-section">
        <h2>AI summary</h2>
        <label class="field"><span>Provider</span>${sel("summary_provider", [["none", "Off"], ["claude", "Claude (Anthropic API)"], ["ollama", "Ollama (local)"]], s.summary_provider)}</label>
        <div class="grid2" data-provider="claude">
          <label class="field"><span>Anthropic API key</span><input type="password" name="anthropic_api_key" value="${esc(s.anthropic_api_key)}" placeholder="sk-ant-…" autocomplete="off"></label>
          <label class="field"><span>Claude model</span><input type="text" name="claude_model" value="${esc(s.claude_model)}"></label>
        </div>
        <div class="grid2" data-provider="ollama">
          <label class="field"><span>Ollama URL</span><input type="url" name="ollama_url" value="${esc(s.ollama_url)}"></label>
          <label class="field"><span>Ollama model</span>${ollama.length ? sel("ollama_model", [["", "Choose…"], ...ollama.map((m) => [m, m])], s.ollama_model) : `<input type="text" name="ollama_model" value="${esc(s.ollama_model)}" placeholder="Ollama not reachable: type a model name">`}</label>
        </div>
        <div class="grid2">
          <label class="field"><span>Summary language</span>${sel("summary_language", [["English", "English"], ["Indonesian", "Indonesian"], ["the same language as the meeting", "Same as the meeting"]], s.summary_language)}</label>
          <label class="field" style="align-self:end"><label class="check"><input type="checkbox" name="auto_summary" ${s.auto_summary ? "checked" : ""}> Summarize automatically after each meeting</label></label>
        </div>
      </div>

      <div class="card settings-section">
        <h2>Automatic recording</h2>
        <label class="check" style="margin-bottom:12px"><input type="checkbox" name="auto_detect_calls" ${s.auto_detect_calls ? "checked" : ""}> Start recording when a call app uses the microphone, stop when the call ends</label>
        <div class="grid2">
          <label class="field"><span>Call apps</span><input type="text" name="auto_detect_apps" value="${esc(s.auto_detect_apps.join(", "))}"><small>Matched against the app or program name.</small></label>
          <label class="field"><span>Also stop after silence (seconds)</span><input type="number" name="auto_stop_silence_s" min="10" value="${s.auto_stop_silence_s}"></label>
        </div>
      </div>

      <div class="settings-actions">
        <button class="btn btn-primary" type="submit">Save settings</button>
        <span class="faint" id="saveNote"></span>
      </div>
    </form>
  </div>`;

  const form = $("#settingsForm");
  const showProvider = () => {
    const p = form.summary_provider.value;
    $$("[data-provider]", form).forEach((el) => { el.hidden = el.dataset.provider !== p; });
  };
  form.summary_provider.addEventListener("change", showProvider);
  showProvider();

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const data = {};
    for (const el of form.elements) {
      if (!el.name) continue;
      data[el.name] = el.type === "checkbox" ? el.checked : el.value;
    }
    try {
      await api("PUT", "/api/settings", data);
      toast("Settings saved");
      refreshStatus();
    } catch (e) { toast(e.message, "error"); }
  });
}

// ------------------------------------------------------------------ upload

$("#uploadBtn").addEventListener("click", () => {
  $("#uploadForm").reset();
  $(".upload-progress").hidden = true;
  $("#uploadDialog").showModal();
});

$("#uploadForm").addEventListener("submit", (ev) => {
  if (ev.submitter?.value !== "ok") return;
  ev.preventDefault();
  const form = ev.target;
  const fd = new FormData(form);
  const prog = $(".upload-progress");
  prog.hidden = false;
  $("#uploadSubmit").disabled = true;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $(".bar span", prog).style.width = `${(e.loaded / e.total) * 100}%`;
    if (e.loaded === e.total) $("small", prog).textContent = "Converting…";
  };
  xhr.onload = () => {
    $("#uploadSubmit").disabled = false;
    if (xhr.status === 200) {
      const meta = JSON.parse(xhr.responseText);
      $("#uploadDialog").close();
      location.hash = `#/m/${meta.id}`;
      refreshMeetings();
    } else {
      let msg = "Upload failed";
      try { msg = JSON.parse(xhr.responseText).detail; } catch { /* keep default */ }
      toast(msg, "error");
      prog.hidden = true;
    }
  };
  xhr.onerror = () => { toast("Upload failed", "error"); $("#uploadSubmit").disabled = false; };
  xhr.send(fd);
});

// ------------------------------------------------------------------ boot

document.addEventListener("click", (e) => {
  const m = $("#menuList");
  if (m && !e.target.closest(".menu")) m.hidden = true;
});

(async function boot() {
  connect();
  await Promise.all([refreshStatus(), refreshMeetings()]);
  route();
  setInterval(refreshStatus, 10000);
})();
