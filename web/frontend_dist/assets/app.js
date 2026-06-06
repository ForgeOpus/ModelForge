const params = new URLSearchParams(location.search);
const urlToken = params.get("token");
if (urlToken) localStorage.setItem("modelforge_token", urlToken);
const token = localStorage.getItem("modelforge_token") || "";

const state = {
  view: "new",
  system: null,
  hardware: null,
  runs: [],
  events: [],
  recommendation: null,
  selectedRun: null,
  error: "",
  form: {
    use_case_slug: "instruction_following",
    objective: "supervised",
    model_id: "",
    dataset_path: "",
    dataset_remote: false,
    output_dir: "./modelforge_output",
  },
};

async function api(path, init = {}) {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-ModelForge-Token": token,
      ...(init.headers || {}),
    },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (value === false || value === null || value === undefined) return;
    if (key === "class") node.className = value;
    else if (key === "onclick") node.onclick = value;
    else if (key === "value") node.value = value;
    else if (key === "checked") node.checked = value;
    else if (key === "selected") node.selected = value;
    else if (key.startsWith("on")) node[key.toLowerCase()] = value;
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.append(typeof child === "string" ? document.createTextNode(child) : child));
  return node;
}

async function refresh() {
  state.system = await api("/api/system");
  state.hardware = await api("/api/hardware");
  state.runs = (await api("/api/runs?limit=50")).runs || [];
  render();
}

function navButton(id, label) {
  return el("button", { class: state.view === id ? "active" : "", onclick: () => { state.view = id; render(); } }, [label]);
}

function render() {
  const root = document.getElementById("app");
  root.innerHTML = "";
  const aside = el("aside", {}, [
    el("div", { class: "brand" }, ["ModelForge"]),
    navButton("new", "New Run"),
    navButton("live", "Live"),
    navButton("history", "History"),
    navButton("models", "Artifacts"),
    navButton("settings", "Settings"),
  ]);
  const workspace = el("section", { class: "workspace" });
  if (state.error) workspace.append(el("div", { class: "error" }, [state.error]));
  if (state.view === "new") renderNew(workspace);
  if (state.view === "live") renderLive(workspace);
  if (state.view === "history") renderHistory(workspace);
  if (state.view === "models") {
    workspace.append(panel("Artifacts", el("p", { class: "muted" }, [
      "Completed output directories appear in history. Hub push actions will attach here as model metadata matures.",
    ])));
  }
  if (state.view === "settings") {
    workspace.append(el("div", { class: "grid two" }, [
      configPanel("System", state.system),
      configPanel("Hardware", state.hardware),
    ]));
  }
  root.append(el("main", { class: "shell" }, [aside, workspace]));
}

function renderNew(workspace) {
  const selected = (state.system?.use_cases || []).find((item) => item.slug === state.form.use_case_slug);
  const form = panel("New fine-tune");
  const useCase = el("select", { onchange: (event) => { state.form.use_case_slug = event.target.value; state.recommendation = null; render(); } });
  (state.system?.use_cases || []).forEach((item) => {
    useCase.append(el("option", { value: item.slug, selected: item.slug === state.form.use_case_slug }, [item.name]));
  });
  const objective = el("select", { onchange: (event) => { state.form.objective = event.target.value; state.recommendation = null; render(); } });
  (selected?.compatible_objectives || ["supervised"]).forEach((item) => {
    objective.append(el("option", { value: item, selected: item === state.form.objective }, [item]));
  });
  form.append(
    label("Use case"),
    useCase,
    el("p", { class: "muted" }, [selected?.description || ""]),
    label("Objective"),
    objective,
    button("Generate recommendation", recommend),
    label("Model"),
    input("model_id", "org/model"),
    label("Dataset"),
    input("dataset_path", "path or HF dataset id"),
    el("label", { class: "inline" }, [
      el("input", { type: "checkbox", checked: state.form.dataset_remote, onchange: (event) => { state.form.dataset_remote = event.target.checked; } }),
      " HuggingFace dataset",
    ]),
    label("Output directory"),
    input("output_dir", "./modelforge_output"),
    button("Start training", startRun, "primary"),
  );
  workspace.append(el("div", { class: "grid two" }, [form, configPanel("Recommendation", state.recommendation)]));
}

function renderLive(workspace) {
  const content = el("section", { class: "panel" });
  const toolbar = el("div", { class: "toolbar" }, [el("h1", {}, ["Live run"])]);
  if (state.selectedRun) toolbar.append(button("Cancel", () => api(`/api/runs/${state.selectedRun.id}/cancel`, { method: "POST" })));
  const log = el("div", { class: "log" }, state.events.length
    ? state.events.map((event) => el("div", {}, [`${event.type}: ${event.message}`]))
    : [el("p", { class: "muted" }, ["No live events yet."])]);
  content.append(toolbar, log);
  workspace.append(content);
}

function renderHistory(workspace) {
  const rows = state.runs.map((run) => el("tr", {}, [
    el("td", {}, [String(run.id)]),
    el("td", {}, [run.status]),
    el("td", {}, [run.model_id]),
    el("td", {}, [run.output_dir]),
  ]));
  workspace.append(panel("Run history", el("table", {}, [el("tbody", {}, rows)])));
}

function panel(title, ...children) {
  return el("section", { class: "panel" }, [el("h1", {}, [title]), ...children]);
}

function configPanel(title, data) {
  return panel(title, data ? el("pre", {}, [JSON.stringify(data, null, 2)]) : el("p", { class: "muted" }, ["Nothing loaded."]));
}

function label(text) {
  return el("label", {}, [text]);
}

function input(key, placeholder) {
  return el("input", {
    value: state.form[key] || "",
    placeholder,
    oninput: (event) => { state.form[key] = event.target.value; },
  });
}

function button(text, onclick, cls = "") {
  return el("button", { class: cls, onclick }, [text]);
}

async function recommend() {
  try {
    state.error = "";
    state.recommendation = await api("/api/recommendations", {
      method: "POST",
      body: JSON.stringify({
        use_case_slug: state.form.use_case_slug,
        objective: state.form.objective,
        n: 5,
      }),
    });
    state.form.model_id = state.recommendation.recommended_models?.[0]?.model_id || state.form.model_id;
    render();
  } catch (err) {
    state.error = String(err);
    render();
  }
}

async function startRun() {
  try {
    state.error = "";
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ ...state.form, recommendation: state.recommendation }),
    });
    state.selectedRun = run;
    state.view = "live";
    subscribe(run.id);
    await refresh();
  } catch (err) {
    state.error = String(err);
    render();
  }
}

function subscribe(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/events?token=${token}`);
  socket.onmessage = (message) => {
    const event = JSON.parse(message.data);
    if (event.type !== "heartbeat") {
      state.events = [event, ...state.events].slice(0, 200);
      render();
    }
  };
}

refresh().catch((err) => {
  state.error = String(err);
  render();
});
