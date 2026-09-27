/* dhaos — interface web (aucune dépendance). Parle à l'API locale avec un jeton Bearer. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  token: null,
  backend: localStorage.getItem("dhaos.backend") || "",
  model: localStorage.getItem("dhaos.model") || "",
  sessionId: null,
  busy: false,
  kbSelected: null,
  config: null,
};

// ------------------------------------------------------------------ utilitaires
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}
function fmtArgs(args) {
  if (!args || typeof args !== "object") return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 80 ? v.slice(0, 80) + "…" : v) : JSON.stringify(v)}`)
    .join(", ");
}
function shortPath(p) {
  const parts = String(p || "").split("/").filter(Boolean);
  return parts.length > 2 ? "…/" + parts.slice(-2).join("/") : String(p || "");
}
function setProjectRoot(p) {
  const el = $("#project-root");
  el.textContent = shortPath(p);
  el.title = `racine du projet : ${p || ""}`;
}
function notice(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle("error", isError);
  el.classList.remove("hidden");
}

// ---------------------------------------------------------------------- API
async function api(method, path, body, opts = {}) {
  const headers = { Authorization: `Bearer ${state.token || ""}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  if (res.status === 401) {
    showTokenModal("Jeton refusé ou absent.");
    throw new Error("jeton d'accès requis");
  }
  if (opts.raw) return res;
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
  return data;
}

// ------------------------------------------------------------ Markdown minimal
function renderInline(text) {
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}
function renderMarkdown(src) {
  const out = [];
  const lines = String(src ?? "").replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  const para = [];
  const flushPara = () => {
    if (para.length) { out.push(`<p>${renderInline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para.length = 0; }
  };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\w+)?\s*$/);
    if (fence) {
      flushPara();
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      const lang = fence[1] ? ` data-lang="${esc(fence[1])}"` : "";
      out.push(`<pre${lang}><button class="btn copy" type="button">copier</button><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushPara(); out.push(`<h${h[1].length}>${renderInline(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushPara(); out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flushPara();
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${renderMarkdown(buf.join("\n"))}</blockquote>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flushPara();
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => renderInline(c.trim()));
      const head = cells(rows[0]);
      const body = rows.slice(2).map((r) => `<tr>${cells(r).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`);
      continue;
    }
    const li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (!m) break;
        let item = m[3];
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !lines[i].match(/^\s*([-*+]|\d+[.)])\s+/)) item += " " + lines[i++].trim();
        items.push(`<li>${renderInline(item)}</li>`);
      }
      out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    if (line.trim() === "") { flushPara(); i++; continue; }
    para.push(line);
    i++;
  }
  flushPara();
  return out.join("\n");
}

// -------------------------------------------------------------------- thème
function applyTheme(t) {
  if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme");
}
applyTheme(localStorage.getItem("dhaos.theme") || "");
$("#theme-toggle").onclick = () => {
  const current = document.documentElement.getAttribute("data-theme");
  const dark = current ? current === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  localStorage.setItem("dhaos.theme", next);
  applyTheme(next);
};

// --------------------------------------------------------------------- jeton
function showTokenModal(err) {
  $("#token-modal").classList.remove("hidden");
  const e = $("#token-error");
  if (err) { e.textContent = err; e.classList.remove("hidden"); } else e.classList.add("hidden");
  $("#token-input").focus();
}
$("#token-form").onsubmit = async (ev) => {
  ev.preventDefault();
  state.token = $("#token-input").value.trim();
  localStorage.setItem("dhaos.token", state.token);
  try {
    await api("GET", "/config");
    $("#token-modal").classList.add("hidden");
    await boot();
  } catch (e) {
    showTokenModal(String(e.message || e));
  }
};
(function readToken() {
  const url = new URL(location.href);
  const t = url.searchParams.get("token");
  if (t) {
    state.token = t;
    localStorage.setItem("dhaos.token", t);
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + (url.search || ""));
  } else {
    state.token = localStorage.getItem("dhaos.token");
  }
})();

// ---------------------------------------------------------------------- tabs
$$(".tab").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "kb") loadBases();
  if (name === "journal") loadJournal();
  if (name === "config") loadConfig();
}

// ------------------------------------------------------------------- santé
async function loadHealth() {
  try {
    const h = await api("GET", "/health");
    const ok = !!(h.backend && h.backend.health && h.backend.health.ok);
    $("#health-dot").className = `health-dot ${ok ? "ok" : "ko"}`;
    const pill = $("#backend-status");
    pill.className = `pill ${ok ? "ok" : "ko"}`;
    pill.textContent = ok ? `${h.backend.name} · ${h.backend.model} · prêt` : `${h.backend.name} indisponible`;
    pill.title = (h.backend.health && h.backend.health.detail) || "";
    $("#version").textContent = `v${h.version}`;
    if (!state.backend) state.backend = h.backend.name;
    $("#backend-select").value = state.backend;
  } catch (e) {
    $("#health-dot").className = "health-dot ko";
  }
}
async function loadModels() {
  const list = $("#model-list");
  list.innerHTML = "";
  try {
    const m = await api("GET", `/models?backend=${encodeURIComponent(state.backend || "")}`);
    (m.models || []).forEach((name) => { const o = document.createElement("option"); o.value = name; list.appendChild(o); });
    if (!state.model) $("#model-input").placeholder = `défaut : ${m.default || ""}`;
  } catch (e) { /* silencieux */ }
}
$("#backend-select").onchange = (ev) => {
  state.backend = ev.target.value;
  localStorage.setItem("dhaos.backend", state.backend);
  state.model = "";
  $("#model-input").value = "";
  localStorage.setItem("dhaos.model", "");
  loadModels();
};
$("#model-input").onchange = (ev) => { state.model = ev.target.value.trim(); localStorage.setItem("dhaos.model", state.model); };

// ------------------------------------------------------------------ sessions
async function loadSessions() {
  const ul = $("#sessions");
  try {
    const items = await api("GET", "/sessions");
    ul.innerHTML = "";
    if (!items.length) { ul.innerHTML = '<li class="muted" style="cursor:default">aucune session</li>'; return; }
    for (const s of items) {
      const li = document.createElement("li");
      li.dataset.id = s.id;
      li.classList.toggle("active", s.id === state.sessionId);
      li.innerHTML = `<span class="title">${esc(s.title || s.id)}</span>
        <span class="meta"><span>${esc(s.backend)} · ${s.n_messages} msg</span><span>${fmtDate(s.updated_at)} <button class="del" title="supprimer">✕</button></span></span>`;
      li.onclick = () => openSession(s.id);
      $(".del", li).onclick = async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Supprimer la session « ${s.title || s.id} » ?`)) return;
        await api("DELETE", `/sessions/${encodeURIComponent(s.id)}`);
        if (state.sessionId === s.id) newChat();
        loadSessions();
      };
      ul.appendChild(li);
    }
  } catch (e) { ul.innerHTML = `<li class="error">${esc(e.message)}</li>`; }
}
$("#refresh-sessions").onclick = loadSessions;

async function openSession(id) {
  if (state.busy) return;
  const s = await api("GET", `/sessions/${encodeURIComponent(id)}`);
  state.sessionId = id;
  const box = $("#messages");
  box.innerHTML = "";
  const pendingTools = {};
  for (const m of s.messages) {
    if (m.role === "user") addUserMessage(m.content);
    else if (m.role === "assistant") {
      const el = addAssistantMessage();
      setAssistantText(el, m.content || "");
      for (const c of m.tool_calls || []) pendingTools[c.id] = addToolCard(c.name, c.arguments);
    } else if (m.role === "tool") {
      const card = pendingTools[m.tool_call_id];
      if (card) setToolResult(card, { is_error: m.is_error, preview: (m.content || "").slice(0, 300) });
    }
  }
  $("#session-label").textContent = `session ${id}`;
  $("#usage-label").textContent = "";
  showTab("chat");
  loadSessions();
  box.scrollTop = box.scrollHeight;
}
function newChat() {
  state.sessionId = null;
  $("#messages").innerHTML = $("#messages").innerHTML.includes("empty-state") ? $("#messages").innerHTML : "";
  if (!$("#messages .empty-state")) $("#messages").innerHTML = '<div class="empty-state"><div class="empty-mark"></div><h2>Nouvelle conversation.</h2><p>Que puis-je faire ?</p></div>';
  $("#session-label").textContent = "nouvelle session";
  $("#usage-label").textContent = "";
  $$("#sessions li").forEach((li) => li.classList.remove("active"));
  $("#prompt").focus();
}
$("#new-chat").onclick = newChat;

// ---------------------------------------------------------------------- chat
function scrollToEnd() { const box = $("#messages"); box.scrollTop = box.scrollHeight; }
function clearEmpty() { const e = $("#messages .empty-state"); if (e) e.remove(); }
function addUserMessage(text) {
  clearEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  $("#messages").appendChild(el);
  scrollToEnd();
  return el;
}
function addAssistantMessage() {
  clearEmpty();
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="avatar"></div><div class="bubble md"></div>`;
  el._raw = "";
  $("#messages").appendChild(el);
  return el;
}
function setAssistantText(el, text, streaming = false) {
  el._raw = text;
  const bubble = $(".bubble", el);
  bubble.innerHTML = renderMarkdown(text);
  bubble.classList.toggle("cursor", streaming);
  $$("pre .copy", bubble).forEach((b) => (b.onclick = () => navigator.clipboard.writeText(b.nextElementSibling.textContent).then(() => (b.textContent = "copié"))));
}
function addToolCard(name, args) {
  const d = document.createElement("details");
  d.className = "tool";
  d.innerHTML = `<summary><span>⚙</span><span class="tname">${esc(name)}</span><span class="targs mono">${esc(fmtArgs(args))}</span><span class="tstatus running">en cours</span></summary><pre class="mono"></pre>`;
  $("#messages").appendChild(d);
  scrollToEnd();
  return d;
}
function setToolResult(card, r) {
  const st = $(".tstatus", card);
  st.textContent = r.is_error ? "erreur" : "ok";
  st.className = `tstatus ${r.is_error ? "error" : "ok"}`;
  $("pre", card).textContent = r.preview || "(vide)";
}
function addConfirmCard(data) {
  const el = document.createElement("div");
  el.className = "confirm";
  el.innerHTML = `<div class="q">Confirmation demandée</div><div class="prompt mono">${esc(data.prompt)}</div>
    <div class="actions"><button class="btn btn-primary yes">Oui, autoriser</button><button class="btn no">Non, refuser</button></div>`;
  const answer = async (ok) => {
    $$(".actions button", el).forEach((b) => (b.disabled = true));
    el.classList.add("answered");
    $(".q", el).textContent = ok ? "Autorisé" : "Refusé";
    try { await api("POST", "/chat/confirm", { id: data.id, answer: ok }); } catch (e) { $(".q", el).textContent = `erreur : ${e.message}`; }
  };
  $(".yes", el).onclick = () => answer(true);
  $(".no", el).onclick = () => answer(false);
  $("#messages").appendChild(el);
  scrollToEnd();
  return el;
}

let activityTimer = null;
function setActivity(label) {
  const box = $("#activity");
  if (!label) { box.classList.add("hidden"); clearInterval(activityTimer); activityTimer = null; return; }
  $("#activity-label").textContent = label;
  box.classList.remove("hidden");
  const t0 = Date.now();
  clearInterval(activityTimer);
  $("#activity-time").textContent = "0 s";
  activityTimer = setInterval(() => ($("#activity-time").textContent = `${Math.round((Date.now() - t0) / 1000)} s`), 1000);
}

async function* readSSE(res) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const data = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) { try { yield { event, data: JSON.parse(data.join("\n")) }; } catch { yield { event, data: {} }; } }
    }
  }
}

async function sendMessage(text) {
  if (state.busy || !text.trim()) return;
  state.busy = true;
  $("#send").disabled = true;
  addUserMessage(text);
  let current = null; // bloc assistant courant, créé au premier texte reçu
  let acc = "";
  const tools = {};
  const ensureBlock = () => { if (!current) current = addAssistantMessage(); return current; };
  setActivity("le modèle lit le contexte et génère…");
  try {
    const body = { message: text, stream: true, no_tools: !$("#tools-toggle").checked };
    if (state.sessionId) body.session_id = state.sessionId;
    if (state.backend) body.backend = state.backend;
    if (state.model) body.model = state.model;
    const res = await api("POST", "/chat", body, { raw: true });
    if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${res.status}`); }
    const sid = res.headers.get("X-Session-Id");
    if (sid) { state.sessionId = sid; $("#session-label").textContent = `session ${sid}`; }
    for await (const { event, data } of readSSE(res)) {
      if (event === "text") {
        acc += data.text;
        setAssistantText(ensureBlock(), acc, true);
        setActivity(null);
        scrollToEnd();
      } else if (event === "thinking") {
        setActivity("le modèle réfléchit…");
      } else if (event === "tool_call") {
        if (current) setAssistantText(current, acc, false);
        tools[data.id] = addToolCard(data.name, data.arguments);
        setActivity(`exécution de ${data.name}…`);
      } else if (event === "tool_result") {
        if (tools[data.id]) setToolResult(tools[data.id], data);
        current = null; // le texte suivant ouvrira un nouveau bloc
        acc = "";
        setActivity("le modèle poursuit…");
      } else if (event === "confirm") {
        addConfirmCard(data);
        setActivity("en attente de votre confirmation…");
      } else if (event === "done") {
        if (current) setAssistantText(current, acc, false);
        const u = data.usage || {};
        $("#usage-label").textContent = `${u.input_tokens || 0} → ${u.output_tokens || 0} jetons · ${data.iterations} itération(s) · ${data.tool_calls} outil(s) · ${data.stop_reason}${data.error ? " · erreur : " + data.error : ""}`;
        if (data.error) { const n = document.createElement("div"); n.className = "notice error"; n.style.marginLeft = "40px"; n.textContent = data.error; $("#messages").appendChild(n); }
      } else if (event === "error") {
        const n = document.createElement("div"); n.className = "notice error"; n.style.marginLeft = "40px"; n.textContent = data.detail || "erreur"; $("#messages").appendChild(n);
      }
    }
  } catch (e) {
    const n = document.createElement("div"); n.className = "notice error"; n.style.marginLeft = "40px"; n.textContent = String(e.message || e); $("#messages").appendChild(n);
  } finally {
    setActivity(null);
    state.busy = false;
    $("#send").disabled = false;
    scrollToEnd();
    loadSessions();
    $("#prompt").focus();
  }
}
$("#composer").onsubmit = (ev) => {
  ev.preventDefault();
  const ta = $("#prompt");
  const text = ta.value;
  ta.value = "";
  ta.style.height = "auto";
  sendMessage(text);
};
$("#prompt").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#composer").requestSubmit(); }
});
$("#prompt").addEventListener("input", (ev) => { const ta = ev.target; ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px"; });

// ------------------------------------------------------------ bases de savoir
async function loadBases() {
  const ul = $("#kb-list");
  try {
    const bases = await api("GET", "/kb");
    ul.innerHTML = "";
    if (!bases.length) ul.innerHTML = '<li class="muted" style="cursor:default">aucune base</li>';
    for (const b of bases) {
      const li = document.createElement("li");
      li.classList.toggle("active", b.name === state.kbSelected);
      li.innerHTML = `<div class="name">${esc(b.name)}</div><div class="meta">${esc(b.description || "—")}</div><div class="meta">${b.n_docs} doc · ${b.n_chunks} chunks · ${esc(b.embedder)}</div>`;
      li.onclick = () => selectBase(b.name);
      ul.appendChild(li);
    }
    if (state.kbSelected && !bases.some((b) => b.name === state.kbSelected)) { state.kbSelected = null; $("#kb-detail").classList.add("hidden"); $("#kb-empty").classList.remove("hidden"); }
  } catch (e) { ul.innerHTML = `<li class="error">${esc(e.message)}</li>`; }
}
$("#refresh-kb").onclick = loadBases;
$("#kb-create").onsubmit = async (ev) => {
  ev.preventDefault();
  try {
    const b = await api("POST", "/kb", { name: $("#kb-new-name").value.trim(), description: $("#kb-new-desc").value.trim() });
    $("#kb-new-name").value = ""; $("#kb-new-desc").value = "";
    await loadBases();
    selectBase(b.name);
  } catch (e) { alert(e.message); }
};
async function selectBase(name) {
  state.kbSelected = name;
  $$("#kb-list li").forEach((li) => li.classList.toggle("active", $(".name", li) && $(".name", li).textContent === name));
  $("#kb-empty").classList.add("hidden");
  $("#kb-detail").classList.remove("hidden");
  $("#kb-add-result").classList.add("hidden");
  $("#kb-results").innerHTML = "";
  await refreshBase();
}
async function refreshBase() {
  const name = state.kbSelected;
  if (!name) return;
  const d = await api("GET", `/kb/${encodeURIComponent(name)}`);
  $("#kb-name").textContent = d.name;
  $("#kb-desc").value = d.description || "";
  const size = (d.documents || []).reduce((a, x) => a + (x.size || 0), 0);
  $("#kb-stats").innerHTML = [["documents", d.n_docs], ["chunks", d.n_chunks], ["embedder", d.embedder], ["taille", `${(size / 1024).toFixed(0)} Ko`]]
    .map(([k, v]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`).join("");
  $("#kb-doc-count").textContent = `(${(d.documents || []).length})`;
  const ul = $("#kb-docs");
  ul.innerHTML = "";
  for (const doc of d.documents || []) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="src mono" title="${esc(doc.source)}">${esc(doc.source)}</span><span class="muted">${doc.n_chunks} chunks</span><button class="icon-btn" title="retirer">✕</button>`;
    $("button", li).onclick = async () => {
      if (!confirm(`Retirer « ${doc.source} » de la base ?`)) return;
      await api("DELETE", `/kb/${encodeURIComponent(name)}/documents`, { source: doc.source });
      refreshBase(); loadBases();
    };
    ul.appendChild(li);
  }
}
$("#kb-desc-save").onclick = async () => {
  try { await api("PATCH", `/kb/${encodeURIComponent(state.kbSelected)}`, { description: $("#kb-desc").value }); loadBases(); }
  catch (e) { alert(e.message); }
};
$("#kb-rename").onclick = async () => {
  const n = prompt("Nouveau nom de la base :", state.kbSelected);
  if (!n || n === state.kbSelected) return;
  try { const b = await api("PATCH", `/kb/${encodeURIComponent(state.kbSelected)}`, { new_name: n }); state.kbSelected = b.name; await loadBases(); selectBase(b.name); }
  catch (e) { alert(e.message); }
};
$("#kb-delete").onclick = async () => {
  if (!confirm(`Supprimer définitivement la base « ${state.kbSelected} » et ses documents ?`)) return;
  await api("DELETE", `/kb/${encodeURIComponent(state.kbSelected)}`);
  state.kbSelected = null;
  $("#kb-detail").classList.add("hidden"); $("#kb-empty").classList.remove("hidden");
  loadBases();
};
$("#kb-reindex").onclick = async () => {
  const out = $("#kb-add-result");
  notice(out, "réindexation en cours…");
  try { const r = await api("POST", `/kb/${encodeURIComponent(state.kbSelected)}/reindex`); notice(out, `réindexé : ${r.chunks} chunk(s)`); refreshBase(); loadBases(); }
  catch (e) { notice(out, e.message, true); }
};
$("#kb-add").onsubmit = async (ev) => {
  ev.preventDefault();
  const sources = $("#kb-sources").value.split(/\s+/).map((s) => s.trim()).filter(Boolean);
  const out = $("#kb-add-result");
  notice(out, "ingestion en cours… (les embeddings peuvent prendre un moment)");
  try {
    const r = await api("POST", `/kb/${encodeURIComponent(state.kbSelected)}/documents`, { sources, recursive: $("#kb-recursive").checked });
    notice(out, r.summary + (r.errors && r.errors.length ? "\n" + r.errors.join("\n") : ""), r.failed > 0 && r.added + r.updated === 0);
    $("#kb-sources").value = "";
    refreshBase(); loadBases();
  } catch (e) { notice(out, e.message, true); }
};
$("#kb-note").onsubmit = async (ev) => {
  ev.preventDefault();
  const out = $("#kb-add-result");
  try {
    const r = await api("POST", `/kb/${encodeURIComponent(state.kbSelected)}/notes`, { text: $("#kb-note-text").value, title: $("#kb-note-title").value || null });
    notice(out, `note ajoutée (document ${r.doc_id})`);
    $("#kb-note-text").value = ""; $("#kb-note-title").value = "";
    refreshBase(); loadBases();
  } catch (e) { notice(out, e.message, true); }
};
$("#kb-search").onsubmit = async (ev) => {
  ev.preventDefault();
  const box = $("#kb-results");
  box.innerHTML = '<div class="muted">recherche…</div>';
  try {
    const hits = await api("POST", "/kb/search", { query: $("#kb-query").value, bases: [state.kbSelected], top_k: Number($("#kb-topk").value) || 5, mode: $("#kb-mode").value });
    box.innerHTML = hits.length ? "" : '<div class="muted">aucun résultat</div>';
    for (const h of hits) {
      const d = document.createElement("div");
      d.className = "hit";
      d.innerHTML = `<div class="src"><span class="mono">${esc(h.source)}</span><span>score ${h.score.toFixed(3)} · chunk ${h.chunk_ord}</span></div><div class="txt">${esc(h.text)}</div>`;
      box.appendChild(d);
    }
  } catch (e) { box.innerHTML = `<div class="notice error">${esc(e.message)}</div>`; }
};

// ------------------------------------------------------------------- journal
async function loadJournal() {
  const tb = $("#journal-table tbody");
  try {
    const rows = await api("GET", "/journal?n=200");
    tb.innerHTML = "";
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="4" class="muted">aucune action journalisée</td></tr>'; return; }
    for (const r of rows.slice().reverse()) {
      const detail = r.command ? r.command : r.path ? r.path : JSON.stringify(r);
      const result = r.kind === "run_command" ? (r.timeout ? "délai dépassé" : r.interrupted ? "interrompu" : `exit ${r.exit}`) + (r.duration != null ? ` · ${r.duration}s` : "")
        : `${r.bytes != null ? r.bytes + " o" : ""}${r.replacements != null ? " · " + r.replacements + " rempl." : ""}${r.backup ? " · sauvegarde" : ""}`;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="muted">${esc(fmtDate(r.iso))}</td><td><code>${esc(r.kind)}</code>${r.confirmed ? ' <span class="pill">confirmé</span>' : ""}</td><td class="mono" style="word-break:break-all">${esc(detail)}</td><td>${esc(result)}</td>`;
      tb.appendChild(tr);
    }
  } catch (e) { tb.innerHTML = `<tr><td colspan="4" class="error">${esc(e.message)}</td></tr>`; }
}
$("#refresh-journal").onclick = loadJournal;

// -------------------------------------------------------------------- config
function getPath(obj, dotted) { return dotted.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj); }
async function loadConfig() {
  try {
    const c = await api("GET", "/config");
    state.config = c;
    $("#config-dump").textContent = JSON.stringify(c, null, 2);
    $("#config-path").textContent = c.source_path || "";
    for (const el of $$("#config-form [name]")) {
      const v = getPath(c, el.name);
      if (el.type === "checkbox") el.checked = !!v; else if (v !== undefined && v !== null) el.value = v;
    }
    setProjectRoot(c.project_root);
  } catch (e) { notice($("#config-result"), e.message, true); }
}
$("#config-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const out = $("#config-result");
  const changes = [];
  for (const el of $$("#config-form [name]")) {
    const value = el.type === "checkbox" ? el.checked : el.value;
    const before = getPath(state.config || {}, el.name);
    if (String(before ?? "") !== String(value)) changes.push({ key: el.name, value });
  }
  if (!changes.length) { notice(out, "aucun changement"); return; }
  try {
    for (const ch of changes) await api("PATCH", "/config", ch);
    notice(out, `enregistré : ${changes.map((c) => c.key).join(", ")}`);
    await loadConfig(); await loadHealth(); await loadModels();
  } catch (e) { notice(out, e.message, true); }
};

// ---------------------------------------------------------------------- boot
async function boot() {
  if (!state.token) { showTokenModal(); return; }
  try { await api("GET", "/config").then((c) => { state.config = c; setProjectRoot(c.project_root); }); }
  catch (e) { return; }
  $("#token-modal").classList.add("hidden");
  $("#backend-select").value = state.backend || "ollama";
  $("#model-input").value = state.model;
  await Promise.all([loadHealth(), loadSessions()]);
  await loadModels();
  $("#prompt").focus();
}
boot();
