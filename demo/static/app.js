"use strict";
/* CountGuard demo UI. Vanilla JS, no dependencies, works offline. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function pct(x, d = 1) {
  if (x == null || isNaN(x)) return "–";
  if (x === 0) return "0%";
  const v = 100 * x;
  if (Math.abs(v) < 0.1) return (v < 0 ? "−" : "") + "<0.1%";
  return v.toFixed(Math.abs(v) >= 10 ? (d > 1 ? 1 : 0) : d).replace("-", "−") + "%";
}
const secs = (s) => (s < 1 ? `${(s * 1000).toFixed(0)} ms` : s < 120 ? `${s.toFixed(s < 10 ? 2 : 1)} s` : `${(s / 60).toFixed(0)} min`);
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
function seg(id, onChange) {
  const el = document.getElementById(id);
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.classList.contains("on")) return;
    $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
    onChange && onChange(b.dataset.v);
  });
  return {
    get: () => $("button.on", el)?.dataset.v,
    set: (v) => $$("button", el).forEach((x) => x.classList.toggle("on", x.dataset.v === String(v))),
    el,
  };
}

/* ------------------------------------------------------------------ nav */
const redrawers = [];
function redrawAll() { redrawers.forEach((f) => f()); }
window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(redrawAll, 120); });

const STEP_INIT = {};

/* ================================================================== frontier chart */
const FAM = {
  suppression: { name: "Suppression", color: "--f-thr", shape: "square" },
  rounding: { name: "Suppression + rounding", color: "--f-rnd", shape: "circle" },
  generalization: { name: "Generalization", slabel: "Generalization: coarser to the right", color: "--f-gen", shape: "diamond" },
  dp: { name: "Differential privacy", color: "--f-dp", shape: "triangle" },
};
const FRONT = {};
async function frontierData(n) {
  if (!FRONT[n]) FRONT[n] = await getJSON(`/api/frontier?n=${n}`);
  return FRONT[n];
}
function shortLabel(p) {
  if (p.group === "rounding") return `round ${p.params.base}`;
  return p.label;
}
/* slider ticks: the slider's own label already names the family */
function tickLabel(p) {
  if (p.group === "generalization") return p.params.short || p.label;
  return shortLabel(p).replace(/^DP /, "");
}
function clusterLabel(list) {
  if (list.length === 1) return shortLabel(list[0]);
  const by = {};
  for (const p of list) (by[p.group] ||= []).push(p);
  const out = [];
  if (by.suppression) out.push("k=" + by.suppression.map((p) => p.params.k).join(", "));
  if (by.rounding) out.push("round " + by.rounding.map((p) => p.params.base).join(", "));
  if (by.generalization) out.push(by.generalization.map(shortLabel).join(" · "));
  if (by.dp) out.push("DP ρ=" + by.dp.map((p) => p.params.rho).join(", "));
  return out.join(" · ");
}

function marker(shape, x, y, r, fill, cls = "mk") {
  const s = `fill="${fill}" class="${cls}"`;
  if (shape === "circle") return `<circle cx="${x}" cy="${y}" r="${r}" ${s}/>`;
  if (shape === "square") return `<rect x="${x - r * 0.9}" y="${y - r * 0.9}" width="${r * 1.8}" height="${r * 1.8}" rx="2" ${s}/>`;
  if (shape === "diamond") return `<path d="M${x} ${y - r * 1.25}L${x + r * 1.25} ${y}L${x} ${y + r * 1.25}L${x - r * 1.25} ${y}Z" ${s}/>`;
  const tri = `M${x} ${y - r * 1.25}L${x + r * 1.15} ${y + r * 0.85}L${x - r * 1.15} ${y + r * 0.85}Z`;
  if (shape === "triangle-open") return `<path d="${tri}" fill="var(--surface)" stroke="${fill}" stroke-width="2.5" class="${cls}"/>`;
  return `<path d="${tri}" ${s}/>`;
}
function mkSwatch(f) {
  const c = css(FAM[f].color);
  return `<svg class="mkr" viewBox="-7 -7 14 14">${marker(FAM[f].shape, 0, 0, 5, c, "")}</svg>`;
}

function drawFrontier(svg, data, { sel = null, onSelect = null, compact = false } = {}) {
  const W = svg.clientWidth || 900, H = svg.clientHeight || 600;
  const m = { l: 64, r: 18, t: 16, b: 52 };
  const pts = data.points;
  const maxL = Math.max(0.12, ...pts.map((p) => p.loss)) * 1.04;
  const maxR = Math.max(0.06, ...pts.map((p) => p.risk)) * 1.06;
  const sx = (v) => m.l + (Math.sqrt(Math.max(0, v)) / Math.sqrt(maxL)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - (Math.sqrt(Math.max(0, v)) / Math.sqrt(maxR)) * (H - m.t - m.b);
  const xt = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1].filter((v) => v <= maxL);
  const yt = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5].filter((v) => v <= maxR);
  let h = "";
  for (const v of xt) h += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${m.t}" y2="${H - m.b}"/><text class="tick" x="${sx(v)}" y="${H - m.b + 18}" text-anchor="middle">${pct(v, 0)}</text>`;
  for (const v of yt) h += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/><text class="tick" x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${pct(v, 0)}</text>`;
  h += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
  h += `<text class="axt" x="${(m.l + W - m.r) / 2}" y="${H - 10}" text-anchor="middle">Accuracy loss</text>`;
  h += `<text class="axt" transform="translate(16 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">Re-identification risk</text>`;
  const bx = sx(data.targets.loss), by = sy(data.targets.risk);
  h += `<rect class="boxr" x="${m.l}" y="${by}" width="${bx - m.l}" height="${H - m.b - by}"/>`;
  // Pareto staircase
  const pp = pts.map((p) => ({ p, r: p.risk, l: p.loss }));
  const par = pp.filter((a) => !pp.some((b) => b !== a && b.r <= a.r && b.l <= a.l && (b.r < a.r || b.l < a.l))).sort((a, b) => a.l - b.l);
  if (par.length > 1) {
    let d = `M${sx(par[0].l)} ${sy(par[0].r)}`;
    for (let i = 1; i < par.length; i++) d += `H${sx(par[i].l)}V${sy(par[i].r)}`;
    h += `<path class="pareto" d="${d}"/>`;
  }
  // markers
  const placed = [{ x: m.l, y: by - 22, w: 330, h: 20 }];  // keep the target-box caption clear
  const R = compact ? 6 : 7;
  for (const p of pts) {
    const x = sx(p.loss), y = sy(p.risk);
    p._x = x; p._y = y;
    placed.push({ x: x - R - 2, y: y - R - 2, w: 2 * R + 4, h: 2 * R + 4 });
  }
  // labels: one per cluster of coincident points, greedy placement, leader line when pushed away
  let lab = "";
  const cl = new Map();
  for (const p of pts) { const key = `${Math.round(p._x / 7)}_${Math.round(p._y / 7)}`; (cl.get(key) || cl.set(key, []).get(key)).push(p); }
  const reps = [...cl.values()].map((list) => Object.assign(Object.create(list[0]), { _text: clusterLabel(list), _sel: list.some((q) => q.id === sel) }));
  const order = reps.sort((a, b) => (b._sel - a._sel) || (b.pareto - a.pareto) || (a.loss - b.loss));
  const fs = compact ? 12 : 12.5, cwid = fs * 0.56;
  const inPlot = (b) => b.x >= m.l + 2 && b.x + b.w <= W - m.r && b.y >= m.t && b.y + b.h <= H - m.b - 2;
  const overlaps = (b) => placed.some((o) => b.x < o.x + o.w && b.x + b.w > o.x && b.y < o.y + o.h && b.y + b.h > o.y);
  for (const p of order) {
    const text = p._text, w = text.length * cwid + 4, hh = fs + 4;
    let best = null;
    outer: for (const rad of [11, 22, 36, 54, 76, 100, 130]) {
      for (const ang of [0, 180, -35, 35, -90, 90, -145, 145, -60, 60, -120, 120, -15, 15, 165, -165]) {
        const a = (ang * Math.PI) / 180;
        const cx = p._x + Math.cos(a) * rad, cy = p._y + Math.sin(a) * rad;
        const bx2 = Math.cos(a) >= 0.2 ? cx : Math.cos(a) <= -0.2 ? cx - w : cx - w / 2;
        const b = { x: bx2, y: cy - hh / 2, w, h: hh };
        if (inPlot(b) && !overlaps(b)) { best = { b, rad }; break outer; }
      }
    }
    if (!best) continue;
    placed.push(best.b);
    const tx = best.b.x + 2, ty = best.b.y + hh / 2 + fs * 0.35;
    if (best.rad > 22) {
      const ex = Math.max(best.b.x, Math.min(p._x, best.b.x + best.b.w)), ey = Math.max(best.b.y, Math.min(p._y, best.b.y + best.b.h));
      lab += `<line class="lead" x1="${p._x}" y1="${p._y}" x2="${ex}" y2="${ey}"/>`;
    }
    lab += `<text class="pl${sel && !p._sel && compact ? " dim" : ""}" x="${tx}" y="${ty}" style="font-size:${fs}px${p._sel ? ";font-weight:700" : ""}">${esc(text)}</text>`;
  }
  let mk = "";
  for (const p of pts) {
    const c = css(FAM[p.group].color);
    if (p.id === sel) mk += `<circle class="halo" cx="${p._x}" cy="${p._y}" r="${R + 6}"/>`;
    mk += `<g data-id="${p.id}">${marker(FAM[p.group].shape, p._x, p._y, R, c, "mk" + (p.id === sel ? " sel" : ""))}<circle cx="${p._x}" cy="${p._y}" r="${R + 8}" fill="transparent" style="cursor:pointer"/></g>`;
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = h + lab + mk;
  if (onSelect) svg.onclick = (e) => { const g = e.target.closest("g[data-id]"); if (g) onSelect(g.dataset.id); };
}

/* ================================================================== STEP 1: attack and protect */
/* One view: the frontier of every release rule, full width, with the selected rule's numbers above.
   Everything is precomputed. */
const FAM_ORDER = ["suppression", "rounding", "generalization", "dp"];
const FAM_SHORT = { suppression: "Suppress", rounding: "Suppress + round", generalization: "Generalize", dp: "DP" };
const S1 = { n: "285", sel: "p1", d: null };

$("#a-fam").innerHTML = FAM_ORDER.map((f) => `<button data-v="${f}"${f === "suppression" ? ' class="on"' : ""}>${FAM_SHORT[f]}</button>`).join("");
const a_n = seg("a-n", (v) => { const lab = cur()?.label; S1.n = v; select(null, lab); });
// family switch lands on: k=10, round 5 (k=10), the first (lightest) generalization, DP ρ=0.1
const a_fam = seg("a-fam", (v) => { const g = S1.d.points.filter((p) => p.group === v); select(g[Math.min(v === "generalization" ? 0 : 1, g.length - 1)].id); });
$("#a-slider").addEventListener("input", (e) => select(S1.d.points.filter((p) => p.group === cur().group)[+e.target.value].id));
$("#a-ticks").addEventListener("click", (e) => { const t = e.target.closest("span[data-i]"); if (t) { $("#a-slider").value = t.dataset.i; $("#a-slider").dispatchEvent(new Event("input")); } });
/* Recommend: the most accurate rule inside the target box, or the closest one if none is inside. */
$("#a-rec").addEventListener("click", () => select(S1.d.recommended.id));
STEP_INIT[1] = () => select(S1.sel);

const cur = () => S1.d && S1.d.points.find((p) => p.id === S1.sel);

/* select a rule by id, or (after a cohort switch) by label */
async function select(id, label) {
  S1.d = await frontierData(S1.n);
  const p = (label && S1.d.points.find((q) => q.label === label)) || S1.d.points.find((q) => q.id === id) || S1.d.points[1];
  S1.sel = p.id; a_fam.set(p.group);
  renderStep1();
}

function renderStep1() {
  const d = S1.d, p = cur();
  if (!p) return;
  const g = d.points.filter((q) => q.group === p.group), idx = g.indexOf(p);
  const sl = $("#a-slider"); sl.max = g.length - 1; sl.value = idx;
  $("#a-slabel").textContent = FAM[p.group].slabel || FAM[p.group].name;
  $("#a-ticks").innerHTML = g.map((q, i) => `<span data-i="${i}" class="${i === idx ? "on" : ""}">${esc(tickLabel(q))}</span>`).join("");
  // generalization is judged on the released (merged) groups, so it has its own count of small groups
  const small = p.small_groups ?? d.cohort.small_cells_1_9;
  $('[data-c="certain"]').innerHTML = `${fmtInt(p.certain)}<span class="of"> / ${fmtInt(small)}</span>`;
  $('[data-c="risk"]').textContent = pct(Math.max(0, p.raw - p.floor));
  // risk = share of the whole cohort sitting in a small group whose size the attacker guesses right,
  // minus the share a blind guess (the same size for every small group) would already get
  const p1 = (x) => Math.round(100 * x) + "%";
  $('[data-c="riskwhy"]').textContent = `${p1(p.raw)} of patients correctly sized − ${p1(p.floor)} by blind guessing`;
  $('[data-c="loss"]').textContent = pct(p.loss);
  drawFrontier($("#a-chart"), d, { sel: p.id, compact: true, onSelect: (id) => select(id) });
  $("#a-flegend").innerHTML = FAM_ORDER.map((f) => `<span>${mkSwatch(f)}${FAM[f].name}</span>`).join("") +
    `<span><svg class="mkr" viewBox="0 0 14 14"><rect x="1.5" y="1.5" width="11" height="11" fill="${css("--box")}" stroke="${css("--box-edge")}" stroke-width="1.5" stroke-dasharray="3 2"/></svg>Target zone</span>` +
    `<span><svg class="mkr" viewBox="0 0 14 14"><path d="M1 12H7V3H13" fill="none" stroke="${css("--ink-2")}" stroke-width="2" stroke-dasharray="2 3"/></svg>Pareto front</span>`;
}
redrawers.push(() => { if (!$("#step1").hidden && S1.d) renderStep1(); });

/* ------------------------------------------------------------------ boot */
if (location.hash) history.replaceState(null, "", location.pathname);
STEP_INIT[1]();
requestAnimationFrame(redrawAll);
