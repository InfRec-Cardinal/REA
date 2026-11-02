const $ = selector => document.querySelector(selector);
const runs = new Map();
const laneNames = { thinking: "S2 THINK", routing: "S1 ROUTE", action: "S0 ACT", episode: "EPISODE", session: "SESSION" };
const statusNames = { queued: "submitted", submitted: "submitted", running: "running", started: "started", completed: "completed", applied: "advice applied", stale: "advice stale", cancelled: "stopped", cancelling: "stopping", failed: "failed", replaying: "replaying" };
const controllerNames = { "sdk-reach": "SDK / direct approach", "vla-detour": "VLA / detour correction", "motion-servo": "Motion controller / final approach" };
let selected = null;
let source;
let events = [];
let intervals = new Map();
let pathPoints = [];
let lastTick = 0;
let origin = performance.now();

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function duration(value) {
  if (value === null || value === undefined) return "—";
  return value < 1000 ? `${value > 0 && value < 10 ? value.toFixed(1) : Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}
function firstDispatch(metrics = {}) {
  const afterReset = Number.isFinite(metrics.firstActionAfterResetMs);
  return {
    value: afterReset ? metrics.firstActionAfterResetMs : metrics.firstActionMs,
    label: afterReset ? "First dispatch" : "First action",
    caption: afterReset ? "after reset" : "from episode start to execution",
    detail: `End-to-end first action: ${duration(metrics.firstActionMs)} · environment reset: ${duration(metrics.resetMs)}`,
  };
}
function fail(error) { $("#error").hidden = false; $("#error").textContent = error.message; }
async function api(path, options) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}
function active() { return ["queued", "running", "cancelling"].includes(selected?.state); }
function elapsed() {
  return active() ? Math.max(events.at(-1)?.atMs ?? 0, performance.now() - origin) : selected?.summary?.durationMs ?? events.at(-1)?.atMs ?? 0;
}

function renderState() {
  if (!selected) return;
  const live = active();
  $("#run-state").textContent = selected.readOnly ? `REPLAY / ${selected.state.toUpperCase()}` : selected.state.toUpperCase();
  $("#run-state").classList.toggle("running", live);
  $("#run-async").disabled = live;
  $("#run-sync").disabled = live;
  $("#cancel").disabled = !live || selected.state === "cancelling" || selected.readOnly;
  $("#source-note").textContent = selected.readOnly ? "Recorded benchmark events · read-only · original timing and results" : `${selected.mode === "sync" ? "Serial baseline" : "Asynchronous"} · local deterministic demo · not a benchmark score`;
  $("#scene-heading").textContent = selected.readOnly ? "BENCHMARK / RECORDED" : "REACH / DETOUR";
  $("#scene").style.display = selected.readOnly ? "none" : "block";
  $("#replay-scene").hidden = !selected.readOnly;
}

function updateScene(transition) {
  const state = (transition?.nextObservation ?? transition?.observation)?.state;
  if (!state?.demonstration) return;
  const x = 25 + state.x * 350;
  const y = 166 - state.y * 370;
  for (const id of ["effector", "effector-halo"]) { $(`#${id}`).setAttribute("cx", x); $(`#${id}`).setAttribute("cy", y); }
  pathPoints.push([x, y]);
  $("#trace").setAttribute("d", pathPoints.map(([px, py], index) => `${index ? "L" : "M"} ${px} ${py}`).join(" "));
}

function ingest(event) {
  if (events.some(existing => existing.sequence === event.sequence)) return;
  events.push(event);
  if (laneNames[event.type] && event.type !== "session" && event.type !== "episode") {
    const key = `${event.type}:${event.id ?? event.step}`;
    if (event.status === "started") intervals.set(key, { ...event, start: event.atMs, end: null });
    if (["completed", "cancelled", "failed"].includes(event.status) && intervals.has(key)) Object.assign(intervals.get(key), { end: event.atMs, result: event.status, candidateId: event.candidateId ?? intervals.get(key).candidateId });
  }
  if (event.type === "transition") updateScene(event.transition);
  if (event.type === "routing" && event.status === "completed") $("#current-controller").textContent = controllerNames[event.candidateId] ?? event.candidateId ?? "—";
  if (event.type === "thinking" && ["completed", "applied", "stale"].includes(event.status)) {
    $("#advice").classList.add("has-advice");
    const previous = events.findLast(item => item.id === event.id && item.status === "completed");
    const text = event.directive ?? previous?.directive ?? event.reason ?? previous?.reason;
    $("#advice p").textContent = `${statusNames[event.status]}${event.step != null ? ` · Step ${event.step}` : ""}${text ? ` — ${typeof text === "string" ? text : JSON.stringify(text)}` : ""}`;
  }
  if (["session", "episode"].includes(event.type) && ["running", "started", "cancelling", "completed", "cancelled", "failed"].includes(event.status)) {
    // A replay retains its own read-only lifecycle; historical events still populate every lane.
    if (!selected.readOnly || ["completed", "cancelled", "failed"].includes(event.status)) selected.state = event.status === "started" ? "running" : event.status;
    if (event.summary) selected.summary = event.summary;
    if (event.error) selected.error = event.error;
    runs.set(selected.id, selected);
    renderState();
    renderRuns();
  }
  if (event.type === "episode" && event.mode) selected.mode = event.mode;
  $("#step-count").textContent = `STEP ${String(event.step ?? events.findLast(item => item.step != null)?.step ?? 0).padStart(2, "0")}`;
  renderEvents();
  renderTimeline();
  renderMetrics();
}

function renderTimeline() {
  const now = elapsed();
  const windowMs = Math.max(6000, Math.ceil(now / 2000) * 2000);
  $("#elapsed").textContent = `${(now / 1000).toFixed(2)} s`;
  $("#ruler-mid").textContent = `${windowMs / 2000} s`;
  $("#ruler-end").textContent = `${windowMs / 1000} s`;
  for (const type of ["thinking", "routing", "action"]) {
    const track = $(`#track-${type}`);
    track.replaceChildren();
    const matching = [...intervals.values()].filter(interval => interval.type === type);
    for (const interval of matching) {
      const live = interval.end === null && active();
      const end = interval.end ?? now;
      const block = element("div", `interval ${live ? "live" : interval.result ?? ""}`);
      const percent = Math.max(0, end - interval.start) / windowMs * 100;
      block.style.left = `${interval.start / windowMs * 100}%`;
      block.style.width = `${percent}%`;
      block.title = `${interval.id}: ${duration(end - interval.start)}${interval.candidateId ? ` · ${interval.candidateId}` : ""}`;
      if (percent > 9) block.textContent = type === "thinking" ? interval.id : interval.candidateId ?? interval.id;
      track.append(block);
    }
    if (type === "action") for (const event of events.filter(item => item.type === "thinking" && item.status === "applied")) {
      const marker = element("span", "advice-marker"); marker.style.left = `${event.atMs / windowMs * 100}%`; marker.title = "System 2 advice applied here"; track.append(marker);
    }
    const working = matching.some(interval => interval.end === null) && active();
    const lane = $(`[data-lane="${type}"]`);
    lane.classList.toggle("active", working);
    lane.querySelector(".lane-state").textContent = working ? { thinking: "thinking", routing: "routing", action: "executing" }[type] : matching.length ? "idle" : "standby";
  }
}

function renderMetrics() {
  const now = elapsed();
  const brain = [...intervals.values()].filter(item => item.type === "thinking");
  const actions = [...intervals.values()].filter(item => item.type === "action");
  let overlap = 0;
  for (const thought of brain) for (const action of actions) overlap += Math.max(0, Math.min(thought.end ?? now, action.end ?? now) - Math.max(thought.start, action.start));
  const metrics = selected?.summary?.asyncMetrics;
  const fallback = {
    firstActionMs: actions[0]?.start,
    overlapMs: overlap,
    blockedThinkingMs: selected?.mode === "sync" ? brain.reduce((sum, item) => sum + (item.end ?? now) - item.start, 0) : 0,
    routedWhileThinking: events.filter(event => event.type === "routing" && event.status === "started" && event.thinking).length,
    corrections: events.filter(event => event.type === "thinking" && event.status === "applied" && event.step > 0).length,
  };
  const values = metrics ?? fallback;
  const first = firstDispatch(values);
  $("#first-action").textContent = duration(first.value);
  $("#first-action-label").textContent = first.label;
  $("#first-action-caption").textContent = first.caption;
  $("#first-action-metric").title = first.detail;
  $("#overlap").textContent = selected ? duration(values.overlapMs) : "—";
  $("#blocked").textContent = selected ? duration(values.blockedThinkingMs) : "—";
  $("#routing-count").textContent = selected ? values.routedWhileThinking ?? 0 : "—";
  $("#corrections").textContent = selected ? values.corrections ?? 0 : "—";
}

function renderEvents() {
  const items = events.filter(event => !["transition", "session"].includes(event.type));
  $("#event-count").textContent = `${items.length} EVENTS`;
  const container = $("#events"); container.replaceChildren();
  for (const event of items.slice(-40).reverse()) {
    const row = element("div", `event ${event.type}`);
    row.append(element("time", "", `${((event.atMs ?? 0) / 1000).toFixed(2)}s`));
    row.append(element("span", "event-type", laneNames[event.type] ?? event.type));
    const detail = element("div", "event-detail");
    detail.append(element("strong", "", `${statusNames[event.status] ?? event.status}${event.candidateId ? ` · ${event.candidateId}` : ""}`));
    if (event.reason || event.error) detail.append(element("div", "", event.reason ?? event.error));
    row.append(detail); container.append(row);
  }
}

function renderRuns() {
  const container = $("#runs");
  if (!runs.size) return;
  container.replaceChildren();
  for (const run of [...runs.values()].reverse()) {
    const button = element("button", `run-row ${selected?.id === run.id ? "selected" : ""}`);
    const label = element("span");
    label.append(element("strong", "", `${run.readOnly ? "Benchmark replay" : run.mode === "sync" ? "Serial baseline" : "Asynchronous"} · ${statusNames[run.state] ?? run.state}`));
    label.append(element("small", "", `${run.summary?.steps ?? "—"} steps · ${run.summary?.reason ?? (run.readOnly ? "recorded events" : "scheduling demo")}`));
    const first = firstDispatch(run.summary?.asyncMetrics);
    const result = element("span", "result", duration(first.value));
    result.title = first.detail;
    result.append(element("small", "", `${first.label} · ${first.caption}`));
    button.append(label, result);
    button.addEventListener("click", () => selectRun(run).catch(fail));
    container.append(button);
  }
}

async function selectRun(run) {
  source?.close();
  selected = { ...run };
  events = []; intervals = new Map(); pathPoints = [];
  origin = performance.now();
  $("#error").hidden = true;
  $("#advice").classList.remove("has-advice");
  $("#advice p").textContent = "System 2 advice appears when ready, marked where it was applied.";
  $("#current-controller").textContent = run.readOnly ? "Reading benchmark events" : "Waiting for the first route";
  updateScene({ observation: { state: { demonstration: true, x: 0.06, y: 0 } } });
  renderState(); renderRuns(); renderTimeline(); renderMetrics();
  source = new EventSource(`/api/runs/${encodeURIComponent(run.id)}/events`);
  source.addEventListener("snapshot", message => {
    const snapshot = JSON.parse(message.data);
    selected = { ...selected, ...snapshot };
    origin = performance.now() - (snapshot.startedAt ? Math.max(0, Date.now() - snapshot.startedAt) : 0);
    $("#connection").textContent = "SSE CONNECTED";
    renderState();
  });
  source.addEventListener("runtime", message => ingest(JSON.parse(message.data)));
  source.onerror = () => { $("#connection").textContent = "RECONNECTING…"; };
}

async function start(mode) {
  $("#run-async").disabled = true; $("#run-sync").disabled = true;
  try {
    const run = await api("/api/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mode }) });
    runs.set(run.id, run);
    await selectRun(run);
  } catch (error) { $("#run-async").disabled = false; $("#run-sync").disabled = false; fail(error); }
}
async function refresh() {
  const result = await api("/api/runs");
  for (const run of result.runs) runs.set(run.id, run);
  renderRuns();
  if (!selected && result.runs.length) await selectRun(result.runs.at(-1));
}
$("#run-async").addEventListener("click", () => start("async"));
$("#run-sync").addEventListener("click", () => start("sync"));
$("#cancel").addEventListener("click", async () => {
  try { Object.assign(selected, await api(`/api/runs/${selected.id}/cancel`, { method: "POST" })); renderState(); } catch (error) { fail(error); }
});
$("#refresh").addEventListener("click", () => refresh().catch(fail));
function tick(time) {
  if (time - lastTick > 100 && active()) { renderTimeline(); renderMetrics(); lastTick = time; }
  requestAnimationFrame(tick);
}
refresh().catch(fail);
requestAnimationFrame(tick);
