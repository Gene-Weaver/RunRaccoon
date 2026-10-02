/* RunRaccoon dashboard. Plain JS, no dependencies, works offline.
 * Data comes from the local server (server.py); charts are drawn as inline SVG. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const SVGNS = "http://www.w3.org/2000/svg";
const ROLE_VAR = { train: "--train", val: "--val", test: "--test" };
// Unpaired metrics get a neutral ink first, so they never read as "train".
const SLOT_VARS = ["--neutral", "--s7", "--s4", "--s5", "--s6", "--s8"];
const DASH = { train: "", val: "6 3", test: "1.5 3.5" };
const STATUS = {
  running: ["●", "running"], finished: ["✓", "finished"], failed: ["✕", "failed"], crashed: ["✕", "crashed"],
  unresponsive: ["❚❚", "unresponsive"], missing: ["?", "folder missing"], unknown: ["?", "unknown"],
};

const state = {
  runs: [], selected: null, tab: "charts", detail: null,
  panelsVersion: null, sections: [], media: {}, mediaStep: {},
  smoothing: 0, chartFilter: "", runFilter: "", collapsed: new Set(), tableView: new Set(), viewer: null,
};

// ------------------------------------------------------------------------------ utilities
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function el(tag, attrs = {}, ...kids) {
  const n = tag.startsWith("svg:") ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.setAttribute("class", v);
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined) n.append(k.nodeType ? k : document.createTextNode(k));
  return n;
}
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}
function fmt(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return String(v);
  if (!isFinite(v)) return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toLocaleString();
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return String(+v.toPrecision(4));
}
function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace("e+", "e");
  if (a >= 100) return Math.round(v).toLocaleString();
  return String(+v.toPrecision(3));
}
function stamp(t) {          // YYYY/MM/DD HH:MM, local time
  if (!t) return "–";
  const d = new Date(t * 1000), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}/${p(d.getMonth() + 1)}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function duration(s) {
  if (s === null || s === undefined) return "";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(sec).padStart(2, "0")}s` : `${sec}s`;
}
function ago(t) {
  if (!t) return "";
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
const isActive = (r) => r && (r.status === "running" || r.status === "unresponsive");
function seriesColor(s, i) { return css(ROLE_VAR[s.role] || SLOT_VARS[Math.min(i, SLOT_VARS.length - 1)]); }
function ema(ys, w) {
  if (!w) return ys;
  let last = 0, n = 0;
  return ys.map((v) => {
    if (v === null || !isFinite(v)) return v;
    n += 1; last = last * w + (1 - w) * v;
    return last / (1 - Math.pow(w, n));
  });
}
function niceTicks(lo, hi, count) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

// ------------------------------------------------------------------------------ run list
function runRow(r) {
  const [endLabel, endValue] = isActive(r) ? ["estEnd:", r.est_end ? stamp(r.est_end) : "after epoch 1"] : ["End:", stamp(r.ended)];
  const sub = [el("b", { text: "Start:" }), ` ${stamp(r.started)}  `, el("b", { text: endLabel }), ` ${endValue}`];
  return el("div", { class: "run-item", "aria-current": r.id === state.selected ? "true" : "false" },
    el("button", { class: "dot-btn", type: "button", "aria-haspopup": "menu", title: `${r.status} - click to star or hide`,
      "aria-label": `${r.name}: ${r.status}. Star or hide`, onclick: (e) => { e.stopPropagation(); openRunMenu(r, e.currentTarget); } },
      el("span", { class: `dot ${r.status}` })),
    el("button", { class: "run-main", type: "button", title: [r.project, r.run_dir].filter(Boolean).join("\n"),
      onclick: () => selectRun(r.id) },
      el("span", { class: "name-row" }, el("span", { class: "name", text: r.name || r.id }),
        r.starred ? el("span", { class: "star", title: "Starred", text: "★" }) : null),
      el("span", { class: "sub" }, sub)));
}

function renderRunList() {
  const nav = $("#run-list");
  const q = state.runFilter.toLowerCase();
  const runs = state.runs.filter((r) => !q || `${r.name} ${r.project} ${r.id} ${r.group || ""}`.toLowerCase().includes(q));
  const shown = runs.filter((r) => !r.hidden), hiddenRuns = runs.filter((r) => r.hidden);
  const active = shown.filter(isActive), past = shown.filter((r) => !isActive(r));
  nav.replaceChildren();
  const group = (label, list) => {
    if (!list.length) return;
    nav.append(el("div", { class: "group-label" }, el("span", { text: label }), el("span", { text: String(list.length) })));
    for (const r of list) nav.append(runRow(r));
  };
  group("Active", active);
  group("Past", past);
  if (!shown.length) nav.append(el("p", { class: "note", style: "padding: 8px", text: state.runs.length ? "No match." : "No runs yet." }));

  const allHidden = state.runs.filter((r) => r.hidden).length;
  $("#hidden-runs").hidden = allHidden === 0;
  const open = $("#hidden-toggle").getAttribute("aria-expanded") === "true";
  $("#hidden-toggle").replaceChildren(el("span", { text: `${open ? "Hide" : "View"} hidden runs` }),
    el("span", { class: "count", text: `${allHidden} ${open ? "▾" : "▴"}` }));
  $("#hidden-list").replaceChildren(...(hiddenRuns.length ? hiddenRuns.map(runRow)
    : [el("p", { class: "note", style: "padding: 8px", text: "No match." })]));
}

function openRunMenu(r, anchor) {
  const menu = $("#run-menu");
  const item = (label, fn) => el("button", { type: "button", role: "menuitem", text: label, onclick: async () => { closeRunMenu(); await fn(); } });
  menu.replaceChildren(
    item(r.starred ? "☆  Unstar" : "★  Star", () => setRunPrefs(r, { starred: !r.starred })),
    item(r.hidden ? "Unhide" : "Hide", () => setRunPrefs(r, { hidden: !r.hidden })));
  const box = anchor.getBoundingClientRect();
  menu.style.left = `${box.left}px`;
  menu.style.top = `${box.bottom + 4}px`;
  menu.hidden = false;
  menu.querySelector("button").focus();
}

function closeRunMenu() { $("#run-menu").hidden = true; }

async function setRunPrefs(r, prefs) {
  Object.assign(r, prefs);                             // optimistic; the next refresh confirms
  renderRunList(); renderHeader();
  await fetch(`/api/runs/${encodeURIComponent(r.id)}/prefs`, {
    method: "POST", headers: { "Content-Type": "application/json", "X-RunRaccoon": "1" }, body: JSON.stringify(prefs) });
}

async function refreshRuns() {
  try {
    state.runs = (await getJSON("/api/runs")).runs;
  } catch { return; }
  if (!state.selected && state.runs.length) {
    const first = state.runs.find(isActive) || state.runs[0];
    selectRun(first.id, true);
  }
  renderRunList();
  const r = currentRun();
  if (r) renderHeader();
}
const currentRun = () => state.runs.find((r) => r.id === state.selected);

// ------------------------------------------------------------------------------ selection
function selectRun(id, replace = false) {
  if (state.selected !== id) {
    if ($("#viewer").open) $("#viewer").close();
    state.selected = id;
    state.detail = null; state.panelsVersion = null; state.sections = []; state.media = {}; state.mediaStep = {};
    state.checkpoint = null; renderCheckpointLegend();
    GIF.cfg = null; GIF.runId = null; clearInterval(GIF.poll);
    $("#sections").replaceChildren(); $("#media").replaceChildren(); $("#plots").replaceChildren(); $("#log").textContent = "";
  }
  const hash = `#run=${encodeURIComponent(id)}&tab=${state.tab}`;
  replace ? history.replaceState(null, "", hash) : history.pushState(null, "", hash);
  $("#empty").hidden = true; $("#run-view").hidden = false;
  renderRunList();
  refreshSelected(true);
}

function setTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const p of document.querySelectorAll(".tab-body")) p.hidden = p.dataset.panel !== tab;
  if (state.selected) history.replaceState(null, "", `#run=${encodeURIComponent(state.selected)}&tab=${tab}`);
  refreshSelected(true);
}

function renderHeader() {
  const r = currentRun();
  if (!r) return;
  $("#run-name").textContent = r.name || r.id;
  $("#run-star").hidden = !r.starred;
  const [icon, label] = STATUS[r.status] || ["?", r.status];
  const st = $("#run-status");
  st.className = `status ${r.status}`;
  st.replaceChildren(el("span", { class: "icon", text: icon, "aria-hidden": "true" }), el("span", { text: label }));
  const runtime = isActive(r) ? Date.now() / 1000 - r.started : r.runtime;
  const items = [
    ["project", r.project], ["id", r.id], ["group", r.group], ["step", r.step],
    ["runtime", duration(runtime)], ["started", r.started ? new Date(r.started * 1000).toLocaleString() : null],
  ];
  const meta = $("#run-meta");
  meta.replaceChildren(...items.filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => el("span", {}, `${k} `, el("b", { text: String(v) }))));
  meta.append(el("span", {
    class: "path", title: "Click to copy", text: r.run_dir,
    onclick: () => navigator.clipboard && navigator.clipboard.writeText(r.run_dir),
  }));
}

async function refreshSelected(force = false) {
  const id = state.selected;
  if (!id) return;
  renderHeader();
  try {
    if (force || !state.detail || isActive(currentRun())) state.detail = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
    if (id !== state.selected) return;
    if (state.tab === "charts") await refreshCharts(id);
    else if (state.tab === "media") await refreshMedia(id);
    else if (state.tab === "plots") renderPlots();
    else if (state.tab === "gifs") { renderGifs(); loadGifControls(); }
    else if (state.tab === "summary") renderKV($("#summary"), state.detail.summary, state.summaryFilter);
    else if (state.tab === "config") renderKV($("#config"), state.detail.config);
    else if (state.tab === "logs") await refreshLog(id);
  } catch (e) { console.warn(e); }
}

// ---------------------------------------------------------------------------------- charts
async function refreshCharts(id) {
  const v = state.panelsVersion === null ? "" : `?v=${state.panelsVersion}`;
  const data = await getJSON(`/api/runs/${encodeURIComponent(id)}/panels${v}`);
  if (id !== state.selected || data.unchanged) return;
  state.panelsVersion = data.version;
  state.sections = data.sections;
  state.checkpoint = data.checkpoint || null;
  renderCheckpointLegend();
  renderSections();
}

function renderCheckpointLegend() {
  const c = state.checkpoint, box = $("#checkpoint-legend");
  box.hidden = !c;
  if (!c) return;
  $("#checkpoint-text").replaceChildren("best checkpoint now: ", el("b", { text: c.where }));
  box.title = `Chosen by ${c.rule} (value ${fmt(c.value)}). Moves whenever a later step becomes the best.`;
}

function renderSections() {
  const root = $("#sections");
  // Reuse the current card width so live refreshes draw off-screen and swap in without a flash.
  const knownWidth = root.querySelector(".card[data-panel]")?.clientWidth || 0;
  const q = state.chartFilter.toLowerCase();
  const frag = document.createDocumentFragment();
  let shown = 0;
  for (const sec of state.sections) {
    const panels = sec.panels.filter((p) => !q || p.title.toLowerCase().includes(q) || p.series.some((s) => s.key.toLowerCase().includes(q)));
    if (!panels.length) continue;
    shown += panels.length;
    const collapsed = state.collapsed.has(sec.name);
    const grid = el("div", { class: "chart-grid" });
    const wrap = el("section", { class: `section${collapsed ? " collapsed" : ""}` },
      el("button", {
        class: "section-head", "aria-expanded": String(!collapsed),
        onclick: () => { state.collapsed.has(sec.name) ? state.collapsed.delete(sec.name) : state.collapsed.add(sec.name); renderSections(); },
      }, el("span", { class: "chev", text: "▾" }), sec.name, el("span", { class: "count", text: String(panels.length) })),
      grid);
    frag.append(wrap);
    for (const p of panels) {
      const card = chartCard(p);
      if (knownWidth) drawCard(card, knownWidth);
      grid.append(card);
    }
  }
  root.replaceChildren(frag);
  if (!shown) root.append(el("p", { class: "note", text: state.sections.length ? "No metric matches the filter." : "No scalar metrics logged yet." }));
  requestAnimationFrame(() => {
    for (const c of root.querySelectorAll(".card[data-panel]")) if (c.clientWidth !== knownWidth) drawCard(c);
  });
}

function chartCard(p) {
  const card = el("div", { class: "card", "data-panel": p.id });
  card._panel = p;
  const note = p.best ? `${p.goal === "min" ? "min" : "max"} ${fmt(p.best.y)} @ ${p.x_label} ${fmt(p.best.x)}` : "";
  const toggle = el("button", {
    class: "tbl-toggle", text: state.tableView.has(p.id) ? "chart" : "table",
    onclick: () => { state.tableView.has(p.id) ? state.tableView.delete(p.id) : state.tableView.add(p.id); renderSections(); },
  });
  card.append(el("div", { class: "card-head" }, el("span", { class: "card-title", title: p.title, text: p.title }),
    el("span", { class: "card-note" }, note, note ? " · " : "", toggle)));
  if (p.series.length > 1) {
    card.append(el("div", { class: "legend" }, p.series.map((s, i) => el("span", {},
      el("span", { class: "swatch", style: `border-color:${seriesColor(s, i)};border-top-style:${s.role === "val" ? "dashed" : s.role === "test" ? "dotted" : "solid"}` }),
      s.label))));
  }
  return card;
}

function drawCard(card, cardWidth) {
  const p = card._panel;
  card.querySelector(":scope > svg, :scope > .tbl-wrap")?.remove();
  if (state.tableView.has(p.id)) return card.append(panelTable(p));
  const width = Math.max(240, (cardWidth || card.clientWidth) - 24), height = 190;
  const m = { l: 46, r: 10, t: 10, b: 24 };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const svg = el("svg:svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${p.title} chart` });

  const series = p.series.map((s, i) => {
    const y = state.smoothing > 0 ? ema(s.y, state.smoothing) : s.y;
    return { ...s, ys: y, color: seriesColor(s, i) };
  });
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.ys.concat(state.smoothing > 0 ? s.y : [])).filter((v) => v !== null && isFinite(v));
  if (!xs.length || !ys.length) { card.append(svg); return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const isLoss = /loss/i.test(p.title);
  const logY = isLoss && y0 > 0 && y1 / y0 > 30;
  const ty = logY ? Math.log10 : (v) => v;
  let ty0 = ty(y0), ty1 = ty(y1);
  if (ty0 === ty1) { ty0 -= Math.abs(ty0) * 0.1 || 1; ty1 += Math.abs(ty1) * 0.1 || 1; }
  const pad = (ty1 - ty0) * 0.06; ty0 -= pad; ty1 += pad;
  const sx = (v) => m.l + ((v - x0) / (x1 - x0)) * iw;
  const sy = (v) => m.t + ih - ((ty(v) - ty0) / (ty1 - ty0)) * ih;

  // grid + axes
  const grid = el("svg:g", { class: "grid" }), axis = el("svg:g", { class: "axis" });
  const yTicks = logY ? niceTicks(Math.ceil(ty0), Math.floor(ty1), 4).filter(Number.isInteger).map((e) => Math.pow(10, e))
    : niceTicks(ty0, ty1, 4);
  for (const t of yTicks) {
    const y = sy(t);
    if (y < m.t - 1 || y > m.t + ih + 1) continue;
    grid.append(el("svg:line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, "stroke-width": 1, "shape-rendering": "crispEdges" }));
    axis.append(el("svg:text", { x: m.l - 6, y: y + 3.5, "text-anchor": "end", text: fmtTick(t) }));
  }
  for (const t of niceTicks(x0, x1, 5)) {
    const x = sx(t);
    axis.append(el("svg:text", { x, y: m.t + ih + 16, "text-anchor": "middle", text: fmtTick(t) }));
  }
  axis.append(el("svg:line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, "stroke-width": 1, "shape-rendering": "crispEdges" }));
  svg.append(grid, axis);
  if (state.checkpoint && p.checkpoint_x !== undefined && p.checkpoint_x >= x0 && p.checkpoint_x <= x1) {
    const cx = sx(p.checkpoint_x);
    svg.append(el("svg:line", { x1: cx, x2: cx, y1: m.t, y2: m.t + ih, stroke: css("--checkpoint"), "stroke-width": 1.5,
      opacity: 0.9, "aria-label": `best checkpoint at ${p.x_label} ${p.checkpoint_x}` }));
  }

  // lines
  const pathOf = (x, y) => {
    let d = "", pen = false;
    for (let i = 0; i < x.length; i++) {
      if (y[i] === null || !isFinite(y[i]) || (logY && y[i] <= 0)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(y[i]).toFixed(1)}`; pen = true;
    }
    return d;
  };
  for (const s of series) {
    if (state.smoothing > 0) svg.append(el("svg:path", { d: pathOf(s.x, s.y), fill: "none", stroke: s.color, "stroke-width": 1, opacity: 0.22 }));
    svg.append(el("svg:path", {
      d: pathOf(s.x, s.ys), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round",
      "stroke-linecap": "round", "stroke-dasharray": DASH[s.role] || null,
    }));
    if (s.x.length <= 12) for (let i = 0; i < s.x.length; i++) {
      if (s.ys[i] === null || !isFinite(s.ys[i])) continue;
      svg.append(el("svg:circle", { cx: sx(s.x[i]), cy: sy(s.ys[i]), r: 3.5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }));
    }
  }
  if (p.best) {
    const s = series.find((q) => q.key === p.best.key);
    if (s) svg.append(el("svg:circle", { cx: sx(p.best.x), cy: sy(p.best.y), r: 4.5, fill: css("--surface"), stroke: s.color, "stroke-width": 2 }));
  }

  // hover layer
  const hover = el("svg:g", { visibility: "hidden" });
  const vline = el("svg:line", { class: "hover-line", y1: m.t, y2: m.t + ih, "stroke-width": 1 });
  hover.append(vline);
  const dots = series.map((s) => { const c = el("svg:circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }); hover.append(c); return c; });
  svg.append(hover);
  const hit = el("svg:rect", { x: m.l, y: 0, width: iw, height: height, fill: "transparent" });
  svg.append(hit);
  const tip = $("#tooltip");
  const nearest = (arr, v) => { let lo = 0, hi = arr.length - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; arr[mid] < v ? (lo = mid) : (hi = mid); } return Math.abs(arr[lo] - v) <= Math.abs(arr[hi] - v) ? lo : hi; };
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * width;
    const xv = x0 + ((px - m.l) / iw) * (x1 - x0);
    let shownX = null;
    const rows = [];
    series.forEach((s, i) => {
      if (!s.x.length) { dots[i].setAttribute("visibility", "hidden"); return; }
      const k = nearest(s.x, xv), yv = s.ys[k];
      if (shownX === null || Math.abs(s.x[k] - xv) < Math.abs(shownX - xv)) shownX = s.x[k];
      if (yv === null || !isFinite(yv)) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("visibility", "visible");
      dots[i].setAttribute("cx", sx(s.x[k])); dots[i].setAttribute("cy", sy(yv));
      rows.push(el("div", { class: "tt-row" }, el("span", {}, el("span", { class: "swatch", style: `border-color:${s.color}` }), s.label), el("b", { text: fmt(yv) })));
    });
    if (shownX === null) return;
    vline.setAttribute("x1", sx(shownX)); vline.setAttribute("x2", sx(shownX));
    hover.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", { class: "tt-x", text: `${p.x_label} ${fmt(shownX)}` }), ...rows);
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${Math.min(window.innerWidth - tw - 8, ev.clientX + 14)}px`;
    tip.style.top = `${ev.clientY + 14}px`;
  });
  hit.addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); tip.hidden = true; });
  card.append(svg);
}

function panelTable(p) {
  const xs = [...new Set(p.series.flatMap((s) => s.x))].sort((a, b) => b - a).slice(0, 200);
  const look = p.series.map((s) => new Map(s.x.map((x, i) => [x, s.y[i]])));
  return el("div", { class: "tbl-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, el("th", { text: p.x_label }), p.series.map((s) => el("th", { text: s.label })))),
    el("tbody", {}, xs.map((x) => el("tr", {}, el("td", { text: fmt(x) }), look.map((m) => el("td", { text: fmt(m.get(x)) })))))));
}

// ----------------------------------------------------------------------------------- media
async function refreshMedia(id) {
  const data = await getJSON(`/api/runs/${encodeURIComponent(id)}/media`);
  if (id !== state.selected) return;
  state.media = data.media;
  renderMedia();
}

function renderMedia() {
  const root = $("#media");
  const keys = Object.keys(state.media).sort();
  if (!keys.length) { root.replaceChildren(el("p", { class: "note", text: "No images logged." })); return; }
  const cards = keys.map((key) => {
    const items = state.media[key];
    const steps = [...new Set(items.map((m) => m.step))].sort((a, b) => a - b);
    const atLatest = state.mediaStep[key] === undefined || state.mediaStep[key] === "latest";
    const step = atLatest ? steps[steps.length - 1] : state.mediaStep[key];
    const current = items.filter((m) => m.step === step);
    const card = el("div", { class: "card media-card" });
    card.append(el("div", { class: "card-head" }, el("span", { class: "card-title", text: key }),
      el("span", { class: "card-note", text: `${steps.length} step${steps.length === 1 ? "" : "s"}` })));
    current.forEach((m, i) => {
      card.append(el("button", { class: "img-btn", title: "Open larger", onclick: () => openViewer({ kind: "media", key, index: i }) },
        el("img", { src: fileUrl(m.path), alt: m.caption || key, loading: "lazy" })));
      if (m.caption) card.append(el("div", { class: "caption", text: m.caption }));
    });
    if (steps.length > 1) {
      const out = el("output", { text: `step ${step}` });
      const slider = el("input", { type: "range", min: 0, max: steps.length - 1, step: 1, value: steps.indexOf(step), "aria-label": `${key} step` });
      slider.addEventListener("input", () => setMediaStep(key, steps, +slider.value));
      card.append(el("div", { class: "media-controls" }, el("span", { text: `${steps[0]}` }), slider, out));
    }
    return card;
  });
  root.replaceChildren(...cards);
  if (state.viewer && state.viewer.kind === "media") renderViewer();
}

const fileUrl = (path, bust) => `/files/${encodeURIComponent(state.selected)}/${path.split("/").map(encodeURIComponent).join("/")}${bust ? `?t=${bust}` : ""}`;

// Shared by the card slider and the viewer slider: the last step is stored as "latest" so live runs keep following new steps.
function setMediaStep(key, steps, i) {
  i = Math.max(0, Math.min(steps.length - 1, i));
  state.mediaStep[key] = i === steps.length - 1 ? "latest" : steps[i];
  renderMedia();
}

function mediaSteps(key) {
  const items = state.media[key] || [];
  const steps = [...new Set(items.map((m) => m.step))].sort((a, b) => a - b);
  const sel = state.mediaStep[key];
  const step = sel === undefined || sel === "latest" || !steps.includes(sel) ? steps[steps.length - 1] : sel;
  return { items, steps, step, current: items.filter((m) => m.step === step) };
}

// ------------------------------------------------------------------------- image viewer
function openViewer(v) {
  state.viewer = v;
  renderViewer();
  const dlg = $("#viewer");
  if (!dlg.open) dlg.showModal();
}

function renderViewer() {
  const v = state.viewer;
  const dlg = $("#viewer");
  if (!v) return;
  const img = $("#viewer-img"), slider = $("#viewer-slider");
  dlg.classList.toggle("plot", v.kind === "plot");
  dlg.classList.toggle("gif", v.kind === "gif");
  if (v.kind === "media") {
    const { steps, step, current } = mediaSteps(v.key);
    if (!current.length) return;
    v.index = Math.min(v.index, current.length - 1);
    const m = current[v.index];
    const si = steps.indexOf(step);
    $("#viewer-title").textContent = v.key;
    $("#viewer-note").textContent = `step ${step} · ${steps.length} step${steps.length === 1 ? "" : "s"}`;
    img.src = fileUrl(m.path);
    img.alt = m.caption || v.key;
    $("#viewer-open").href = fileUrl(m.path);
    $("#viewer-caption").textContent = m.caption || "";
    $("#viewer-pager").hidden = current.length < 2;
    $("#viewer-index").textContent = `image ${v.index + 1} of ${current.length}`;
    $("#viewer-iprev").disabled = v.index === 0;
    $("#viewer-inext").disabled = v.index >= current.length - 1;
    $("#viewer-controls").hidden = steps.length < 2;
    slider.max = String(steps.length - 1);
    slider.value = String(si);
    slider.setAttribute("aria-label", `${v.key} step`);
    $("#viewer-step").textContent = `step ${step}`;
    $("#viewer-prev").disabled = si <= 0;
    $("#viewer-next").disabled = si >= steps.length - 1;
    for (const n of [steps[si - 1], steps[si + 1]]) {          // preload neighbors so scrubbing is instant
      const nm = n === undefined ? null : mediaStepItems(v.key, n)[v.index];
      if (nm) new Image().src = fileUrl(nm.path);
    }
  } else {
    const plots = (state.detail && state.detail[v.kind === "gif" ? "gifs" : "plots"]) || [];
    if (!plots.length) return;
    v.index = Math.max(0, Math.min(v.index, plots.length - 1));
    const p = plots[v.index];
    const src = fileUrl(p.path, Math.floor(p.mtime));
    const noun = v.kind === "gif" ? "GIF" : "figure";
    $("#viewer-title").textContent = p.path.replace(/^plots\/(gifs\/)?/, "");
    $("#viewer-note").textContent = `${noun} ${v.index + 1} of ${plots.length} · ${new Date(p.mtime * 1000).toLocaleTimeString()}`;
    img.src = src;
    img.alt = p.path;
    $("#viewer-open").href = src;
    $("#viewer-caption").textContent = "";
    $("#viewer-pager").hidden = true;
    $("#viewer-controls").hidden = plots.length < 2;
    slider.max = String(plots.length - 1);
    slider.value = String(v.index);
    slider.setAttribute("aria-label", noun);
    $("#viewer-step").textContent = `${noun} ${v.index + 1}`;
    $("#viewer-prev").disabled = v.index === 0;
    $("#viewer-next").disabled = v.index >= plots.length - 1;
  }
}

const mediaStepItems = (key, step) => (state.media[key] || []).filter((m) => m.step === step);

function viewerStep(delta, absolute) {
  const v = state.viewer;
  if (!v) return;
  if (v.kind === "media") {
    const { steps, step } = mediaSteps(v.key);
    setMediaStep(v.key, steps, absolute !== undefined ? absolute : steps.indexOf(step) + delta);
  } else {
    v.index = absolute !== undefined ? absolute : v.index + delta;
    renderViewer();
  }
}

function viewerImage(delta) {
  const v = state.viewer;
  if (!v || v.kind !== "media") return;
  const n = mediaSteps(v.key).current.length;
  v.index = Math.max(0, Math.min(n - 1, v.index + delta));
  renderViewer();
}

function initViewer() {
  const dlg = $("#viewer");
  $("#viewer-close").addEventListener("click", () => dlg.close());
  $("#viewer-prev").addEventListener("click", () => viewerStep(-1));
  $("#viewer-next").addEventListener("click", () => viewerStep(1));
  $("#viewer-iprev").addEventListener("click", () => viewerImage(-1));
  $("#viewer-inext").addEventListener("click", () => viewerImage(1));
  $("#viewer-slider").addEventListener("input", (e) => viewerStep(0, +e.target.value));
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });      // click on the backdrop
  dlg.addEventListener("close", () => { state.viewer = null; $("#viewer-img").removeAttribute("src"); });
  dlg.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && (e.key === "ArrowLeft" || e.key === "ArrowRight")) return;   // slider handles its own
    const map = { ArrowLeft: () => viewerStep(-1), ArrowRight: () => viewerStep(1), ArrowUp: () => viewerImage(-1), ArrowDown: () => viewerImage(1) };
    if (map[e.key]) { e.preventDefault(); map[e.key](); }
  });
}

// ----------------------------------------------------------------------- plots, kv, logs
function renderPlots() {
  const root = $("#plots");
  const plots = (state.detail && state.detail.plots) || [];
  if (!plots.length) { root.replaceChildren(el("p", { class: "note", text: "No figures rendered yet - progress.png appears after the first epoch." })); return; }
  root.replaceChildren(...plots.map((p, i) => {
    const src = fileUrl(p.path, Math.floor(p.mtime));
    return el("div", { class: "card plot-card" },
      el("div", { class: "card-head" }, el("span", { class: "card-title", text: p.path.replace(/^plots\//, "") }),
        el("span", { class: "card-note", text: new Date(p.mtime * 1000).toLocaleTimeString() })),
      el("button", { class: "img-btn", title: "Open larger", onclick: () => openViewer({ kind: "plot", index: i }) },
        el("img", { src, alt: p.path, loading: "lazy" })));
  }));
  if (state.viewer && state.viewer.kind === "plot") renderViewer();
}

// ------------------------------------------------------------------- GIF controller
const GIF = { cfg: null, opts: null, style: null, runId: null, previewTimer: null, poll: null };
const GIF_DEFAULTS = { seconds_per_frame: 0.25, orientation: "landscape", resolution: 1080, hold_last_s: 1.5, panels: true, label: true };
const STYLE_DEFAULTS = { line_width: 2, point_size: 4, opacity: 0.95, fill_opacity: 0.3, show_gt: true, show_pred: true,
                         show_labels: true, halo: true, colors: {}, hidden: [] };

async function loadGifControls(force = false) {
  const id = state.selected;
  if (!id || (!force && GIF.runId === id && GIF.cfg)) return renderGifControls();
  GIF.cfg = await getJSON(`/api/runs/${encodeURIComponent(id)}/gifs/config`);
  if (id !== state.selected) return;
  GIF.runId = id;
  GIF.opts = { ...GIF.cfg.options };
  GIF.style = JSON.parse(JSON.stringify(GIF.cfg.style));
  renderGifControls();
  if (GIF.cfg.job && GIF.cfg.job.state === "running") pollGifJob();
}

function gifRequest() {
  const o = GIF.opts;
  return { options: { seconds_per_frame: o.seconds_per_frame, orientation: o.orientation, resolution: o.resolution,
                      hold_last_s: o.hold_last_s, panels: o.panels, label: o.label },
           style: GIF.cfg.restylable.length ? GIF.style : null };
}

function schedulePreview() {
  clearTimeout(GIF.previewTimer);
  GIF.previewTimer = setTimeout(() => {
    const img = $("#gif-preview");
    if (!img) return;
    const r = gifRequest();
    img.classList.add("loading");
    img.src = `/api/runs/${encodeURIComponent(state.selected)}/gifs/preview?options=${encodeURIComponent(JSON.stringify({ ...r.options, style: r.style }))}`;
  }, 350);
}

function control(label, input, value) {
  return el("label", { class: "gc-field" }, el("span", { class: "gc-label", text: label }), input, value || "");
}

function rangeCtl(label, get, set, min, max, step, unit = "") {
  const out = el("output", { class: "gc-value", text: `${get()}${unit}` });
  const input = el("input", { type: "range", min, max, step, value: get(), "aria-label": label });
  input.addEventListener("input", () => { set(+input.value); out.textContent = `${input.value}${unit}`; schedulePreview(); });
  return control(label, input, out);
}

function checkCtl(label, get, set) {
  const input = el("input", { type: "checkbox" });
  input.checked = !!get();
  input.addEventListener("change", () => { set(input.checked); schedulePreview(); });
  return el("label", { class: "gc-check" }, input, label);
}

function renderGifControls() {
  const box = $("#gif-controls");
  const cfg = GIF.cfg;
  if (!cfg) { box.replaceChildren(el("p", { class: "note", text: "Loading GIF settings…" })); return; }
  if (!cfg.keys.length) { box.hidden = true; return; }
  box.hidden = false;
  const o = GIF.opts, st = GIF.style;
  const seg = el("div", { class: "gc-seg", role: "group", "aria-label": "Orientation" },
    ["landscape", "portrait"].map((v) => el("button", {
      type: "button", "aria-pressed": String(o.orientation === v), text: v === "landscape" ? "Landscape" : "Portrait",
      onclick: () => { o.orientation = v; renderGifControls(); schedulePreview(); },
    })));
  const res = el("select", { "aria-label": "Resolution" }, [720, 1080, 1440, 2160].map((r) =>
    el("option", { value: r, selected: o.resolution === r, text: `${r}p` })));
  res.addEventListener("change", () => { o.resolution = +res.value; schedulePreview(); });

  const general = el("div", { class: "gc-row" },
    rangeCtl("Speed", () => o.seconds_per_frame, (v) => (o.seconds_per_frame = v), 0.05, 1.5, 0.05, " s/frame"),
    rangeCtl("Hold last frame", () => o.hold_last_s, (v) => (o.hold_last_s = v), 0, 5, 0.25, " s"),
    control("Orientation", seg), control("Resolution", res),
    checkCtl("Per-panel GIFs", () => o.panels, (v) => (o.panels = v)),
    checkCtl("Title & progress bar", () => o.label, (v) => (o.label = v)));

  let styleRow;
  if (cfg.restylable.length) {
    const colorsBtn = el("button", { type: "button", class: "gc-btn", "aria-expanded": "false",
      onclick: (e) => toggleColors(e.currentTarget) },
      el("span", { class: "gc-swatches" }, cfg.features.slice(0, 6).map((f) =>
        el("span", { style: `background:${st.colors[f.id] || f.color}` }))),
      `Colors (${cfg.features.length})`);
    st.hidden = st.hidden || [];
    const nShown = cfg.features.filter((f) => !st.hidden.includes(f.id)).length;
    const showBtn = el("button", { type: "button", class: "gc-btn", "aria-expanded": "false",
      onclick: (e) => toggleShown(e.currentTarget) }, `Show (${nShown}/${cfg.features.length})`);
    styleRow = el("div", { class: "gc-row" },
      rangeCtl("Line width", () => st.line_width, (v) => (st.line_width = v), 0.5, 8, 0.5, " px"),
      rangeCtl("Point size", () => st.point_size, (v) => (st.point_size = v), 1, 16, 0.5, " px"),
      rangeCtl("Opacity", () => st.opacity, (v) => (st.opacity = v), 0.1, 1, 0.05),
      rangeCtl("Mask fill", () => st.fill_opacity, (v) => (st.fill_opacity = v), 0, 0.8, 0.05),
      checkCtl("Ground truth", () => st.show_gt, (v) => (st.show_gt = v)),
      checkCtl("Predictions", () => st.show_pred, (v) => (st.show_pred = v)),
      checkCtl("Labels", () => st.show_labels, (v) => (st.show_labels = v)),
      checkCtl("Dark outline", () => st.halo, (v) => (st.halo = v)),
      colorsBtn, showBtn);
  } else {
    styleRow = el("p", { class: "note gc-note", text: "Overlays in these images were drawn by the training script, so only speed, orientation and size can change. Line, opacity and color controls apply to RunRaccoon's QC contact sheets (Ultralytics runs)." });
  }

  const job = (cfg.job && cfg.job.state === "running") ? cfg.job : null;
  const go = el("button", { type: "button", class: "gc-primary", disabled: !!job, text: job ? "Regenerating…" : "Regenerate GIFs",
    onclick: startGifJob });
  const reset = el("button", { type: "button", class: "gc-btn", text: "Reset to defaults", title: "Default speed, size and overlay style (then Regenerate)",
    onclick: () => { GIF.opts = { ...GIF.opts, ...GIF_DEFAULTS }; GIF.style = JSON.parse(JSON.stringify(STYLE_DEFAULTS)); renderGifControls(); } });
  const status = el("div", { class: "gc-status", id: "gif-status" });
  box.replaceChildren(
    el("div", { class: "gc-head" }, el("span", { class: "card-title", text: "GIF settings" }),
      el("span", { class: "card-note", text: cfg.restylable.length ? `restylable: ${cfg.restylable.join(", ")}` : cfg.keys.join(", ") })),
    el("div", { class: "gc-body" },
      el("div", { class: "gc-form" }, general, styleRow, el("div", { class: "gc-actions" }, go, reset, status)),
      el("figure", { class: "gc-preview" }, el("img", { id: "gif-preview", alt: "Preview of the last frame with these settings",
        onload: (e) => e.target.classList.remove("loading") }), el("figcaption", { text: "preview · last frame" }))));
  updateGifStatus(cfg.job);
  schedulePreview();
}

function toggleColors(btn) {
  const existing = $("#gc-colors");
  if (existing) { existing.remove(); btn.setAttribute("aria-expanded", "false"); return; }
  $("#gc-shown")?.remove();
  btn.setAttribute("aria-expanded", "true");
  const st = GIF.style;
  const pop = el("div", { id: "gc-colors", class: "gc-colors", role: "dialog", "aria-label": "Overlay colors" },
    el("div", { class: "gc-colors-grid" }, GIF.cfg.features.map((f) => {
      const input = el("input", { type: "color", value: st.colors[f.id] || f.color, "aria-label": f.label });
      input.addEventListener("input", () => { st.colors[f.id] = input.value; schedulePreview(); });
      return el("label", { class: "gc-color" }, input, el("span", { text: f.label }));
    })),
    el("div", { class: "gc-colors-foot" },
      el("button", { type: "button", class: "gc-btn", text: "Reset colors",
        onclick: () => { st.colors = {}; $("#gc-colors").remove(); renderGifControls(); } }),
      el("button", { type: "button", class: "gc-btn", text: "Done", onclick: () => { $("#gc-colors").remove(); renderGifControls(); } })));
  btn.closest(".gc-row").append(pop);
}

function toggleShown(btn) {
  const existing = $("#gc-shown");
  if (existing) { existing.remove(); btn.setAttribute("aria-expanded", "false"); return; }
  $("#gc-colors")?.remove();
  btn.setAttribute("aria-expanded", "true");
  const st = GIF.style;
  const setAll = (on) => { st.hidden = on ? [] : GIF.cfg.features.map((f) => f.id); $("#gc-shown").remove(); renderGifControls(); schedulePreview(); };
  const pop = el("div", { id: "gc-shown", class: "gc-colors", role: "dialog", "aria-label": "Which overlays are drawn" },
    el("div", { class: "gc-colors-grid" }, GIF.cfg.features.map((f) => {
      const input = el("input", { type: "checkbox" });
      input.checked = !st.hidden.includes(f.id);
      input.addEventListener("change", () => {
        st.hidden = input.checked ? st.hidden.filter((h) => h !== f.id) : [...new Set([...st.hidden, f.id])];
        btn.lastChild.textContent = `Show (${GIF.cfg.features.length - st.hidden.length}/${GIF.cfg.features.length})`;
        schedulePreview();
      });
      return el("label", { class: "gc-color gc-toggle" }, input,
        el("span", { class: "gc-dot", style: `background:${st.colors[f.id] || f.color}` }), el("span", { text: f.label }));
    })),
    el("div", { class: "gc-colors-foot" },
      el("button", { type: "button", class: "gc-btn", text: "Show all", onclick: () => setAll(true) }),
      el("button", { type: "button", class: "gc-btn", text: "Hide all", onclick: () => setAll(false) }),
      el("button", { type: "button", class: "gc-btn", text: "Done", onclick: () => { $("#gc-shown").remove(); renderGifControls(); } })));
  btn.closest(".gc-row").append(pop);
}

function updateGifStatus(job) {
  const s = $("#gif-status");
  if (!s) return;
  if (!job) { s.replaceChildren(); return; }
  if (job.state === "running") {
    const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
    s.replaceChildren(el("div", { class: "gc-bar" }, el("span", { style: `width:${pct}%` })),
      el("span", { text: job.total ? `${job.done} of ${job.total} GIFs` : "starting…" }));
  } else if (job.state === "done") {
    s.replaceChildren(el("span", { text: `✓ ${job.written} GIFs regenerated` }));
  } else if (job.state === "error") {
    s.replaceChildren(el("span", { class: "gc-error", text: `✕ ${job.error}` }));
  }
}

async function startGifJob() {
  const id = state.selected;
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}/gifs/render`, {
    method: "POST", headers: { "Content-Type": "application/json", "X-RunRaccoon": "1" }, body: JSON.stringify(gifRequest()) });
  const data = await r.json();
  GIF.cfg.job = data.job;
  renderGifControls();
  pollGifJob();
}

function pollGifJob() {
  clearInterval(GIF.poll);
  GIF.poll = setInterval(async () => {
    const id = state.selected;
    const cfg = await getJSON(`/api/runs/${encodeURIComponent(id)}/gifs/config`);
    if (id !== state.selected) return clearInterval(GIF.poll);
    GIF.cfg.job = cfg.job;
    updateGifStatus(cfg.job);
    if (!cfg.job || cfg.job.state !== "running") {
      clearInterval(GIF.poll);
      const btn = $(".gc-primary");
      if (btn) { btn.disabled = false; btn.textContent = "Regenerate GIFs"; }
      state.detail = await getJSON(`/api/runs/${encodeURIComponent(id)}`);
      renderGifs();
    }
  }, 1000);
}

function renderGifs() {
  const root = $("#gifs");
  const gifs = (state.detail && state.detail.gifs) || [];
  if (!gifs.length) {
    root.replaceChildren(el("p", { class: "note", text: "No GIFs yet - they are made when a run finishes, from images logged at several steps (or run `runraccoon gif <run dir>`)." }));
    return;
  }
  // Keep cards (and their playing GIFs) when nothing changed, so live refreshes don't restart them.
  const sig = gifs.map((g) => `${g.path}@${Math.floor(g.mtime)}`).join("|");
  if (root.dataset.sig === sig) return;
  root.dataset.sig = sig;
  root.replaceChildren(...gifs.map((g, i) => {
    const name = g.path.replace(/^plots\/gifs\//, "");
    return el("div", { class: "card plot-card gif-card" },
      el("div", { class: "card-head" }, el("span", { class: "card-title", title: name, text: name }),
        el("span", { class: "card-note", text: `${(g.size / 1e6).toFixed(1)} MB` })),
      el("button", { class: "img-btn", title: "Open larger", onclick: () => openViewer({ kind: "gif", index: i }) },
        el("img", { src: fileUrl(g.path, Math.floor(g.mtime)), alt: name, loading: "lazy" })));
  }));
}

function renderKV(table, obj, filter = "") {
  const q = (filter || "").toLowerCase();
  const entries = Object.entries(obj || {}).filter(([k]) => !k.startsWith("_") && (!q || k.toLowerCase().includes(q)))
    .sort(([a], [b]) => a.localeCompare(b));
  table.replaceChildren(...(entries.length ? entries.map(([k, v]) => el("tr", {}, el("td", { text: k }),
    el("td", { text: typeof v === "object" && v !== null ? JSON.stringify(v) : fmt(v) })))
    : [el("tr", {}, el("td", { class: "note", text: "Nothing here." }))]));
}

async function refreshLog(id) {
  const data = await getJSON(`/api/runs/${encodeURIComponent(id)}/log?lines=600`);
  if (id !== state.selected) return;
  const pre = $("#log");
  const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  pre.textContent = data.lines.join("\n") || "(output.log is empty)";
  if (stick) pre.scrollTop = pre.scrollHeight;
}

// ------------------------------------------------------------------------------------ boot
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { run: h.get("run"), tab: h.get("tab") };
}

function init() {
  initViewer();
  $("#hidden-toggle").addEventListener("click", () => {
    const b = $("#hidden-toggle"), open = b.getAttribute("aria-expanded") !== "true";
    b.setAttribute("aria-expanded", String(open));
    $("#hidden-list").hidden = !open;
    $("#hidden-runs").classList.toggle("open", open);
    renderRunList();
  });
  document.addEventListener("click", (e) => { if (!$("#run-menu").hidden && !e.target.closest("#run-menu")) closeRunMenu(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeRunMenu(); });
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setTab(b.dataset.tab));
  $("#run-filter").addEventListener("input", (e) => { state.runFilter = e.target.value; renderRunList(); });
  $("#chart-filter").addEventListener("input", (e) => { state.chartFilter = e.target.value; renderSections(); });
  $("#summary-filter").addEventListener("input", (e) => { state.summaryFilter = e.target.value; renderKV($("#summary"), state.detail && state.detail.summary, e.target.value); });
  const sm = $("#smoothing");
  sm.addEventListener("input", () => {
    state.smoothing = +sm.value; $("#smoothing-out").textContent = sm.value;
    for (const c of document.querySelectorAll(".card[data-panel]")) drawCard(c);
  });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { for (const c of document.querySelectorAll(".card[data-panel]")) drawCard(c); }, 150); });
  window.addEventListener("popstate", () => { const h = readHash(); if (h.tab) setTab(h.tab); if (h.run) selectRun(h.run, true); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => renderSections());

  const h = readHash();
  if (h.tab && document.querySelector(`.tabs button[data-tab="${h.tab}"]`)) setTab(h.tab);
  if (h.run) selectRun(h.run, true);
  refreshRuns();
  setInterval(refreshRuns, 4000);
  setInterval(() => {
    if (document.hidden) return;
    const r = currentRun();
    if (isActive(r)) refreshSelected();
  }, 3000);
}
init();
