// Dengue risk map frontend. Plain Leaflet, data from the FastAPI backend.
const API = window.DENGUE_API || "";

const VIEWS = {
  brazil: {
    center: [-14.5, -52], zoom: 4,
    layers: [["pred", "Modeled incidence per 100k (month)"], ["obs", "Reported incidence per 100k (month)"]],
    bins: [0.5, 2, 5, 10, 25, 50, 100, 250],
    unit: "per 100k",
  },
  india_districts: {
    center: [22.5, 80], zoom: 5,
    layers: [["rel", "Modeled relative risk (percentile among districts)"]],
    bins: [10, 20, 30, 40, 50, 60, 70, 80, 90],
    unit: "percentile",
  },
  india_states: {
    center: [22.5, 80], zoom: 5,
    layers: [["pred_rank", "Modeled incidence, rank percentile"], ["obs_rank", "Reported incidence, rank percentile"]],
    bins: [10, 20, 30, 40, 50, 60, 70, 80, 90],
    unit: "rank percentile",
  },
};
const COLORS = ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603", "#7f2704", "#5a1a02"];

const state = { view: "brazil", layer: "pred", periods: [], values: {}, geo: {}, meta: null };
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.5 }).setView(VIEWS.brazil.center, VIEWS.brazil.zoom);
L.tileLayer("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO. Boundaries: IBGE, DataMeet (Survey of India based).",
  maxZoom: 10,
}).addTo(map);
let geoLayer = null;
let selected = null;

const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => { const d = L.DomUtil.create("div", "legend"); d.id = "legend"; return d; };
legend.addTo(map);

async function getJSON(url) {
  const r = await fetch(API + url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

function color(v, bins) {
  if (v === null || v === undefined || Number.isNaN(v)) return "#cccccc";
  let i = 0;
  while (i < bins.length && v > bins[i]) i++;
  return COLORS[Math.min(i, COLORS.length - 1)];
}

function rankPct(vals, key) {
  const ids = Object.keys(vals).filter((k) => vals[k][key] !== null);
  ids.sort((a, b) => vals[a][key] - vals[b][key]);
  const out = {};
  ids.forEach((id, i) => { out[id] = ids.length > 1 ? (100 * i) / (ids.length - 1) : 50; });
  return out;
}

function valueFor(id) {
  const v = state.values[id];
  if (v === undefined) return null;
  if (state.view === "brazil") return state.layer === "pred" ? v[0] : v[1];
  if (state.view === "india_districts") return v[0];
  return state.ranks[state.layer === "pred_rank" ? "pred" : "obs"][id] ?? null;
}

function renderLegend() {
  const cfg = VIEWS[state.view];
  const b = cfg.bins;
  let html = `<b>${cfg.layers.find((l) => l[0] === state.layer)[1]}</b><br>`;
  html += `<i style="background:${COLORS[0]}"></i> &le; ${b[0]}<br>`;
  for (let i = 1; i < b.length; i++) html += `<i style="background:${COLORS[i]}"></i> ${b[i - 1]} to ${b[i]}<br>`;
  html += `<i style="background:${COLORS[b.length]}"></i> &gt; ${b[b.length - 1]}<br>`;
  html += `<i style="background:#cccccc"></i> no data`;
  document.getElementById("legend").innerHTML = html;
}

function style(f) {
  const v = valueFor(f.properties.region_id);
  const sel = selected === f.properties.region_id;
  return {
    fillColor: color(v, VIEWS[state.view].bins), fillOpacity: 0.85,
    color: sel ? "#000" : "#ffffff", weight: sel ? 2 : state.view === "brazil" ? 0.2 : 0.6,
  };
}

async function loadGeo(view) {
  if (!state.geo[view]) state.geo[view] = await getJSON(`/geo/${view}`);
  if (geoLayer) map.removeLayer(geoLayer);
  geoLayer = L.geoJSON(state.geo[view], {
    style,
    onEachFeature: (f, layer) => {
      layer.on("click", () => { selected = f.properties.region_id; geoLayer.setStyle(style); showDetails(selected); });
      layer.bindTooltip(() => {
        const v = valueFor(f.properties.region_id);
        return `${f.properties.name || f.properties.state}: ${v === null ? "no data" : v.toFixed(1)}`;
      }, { sticky: true });
    },
  }).addTo(map);
}

function period() { return state.periods[+document.getElementById("month").value]; }

async function loadValues() {
  const p = period();
  document.getElementById("monthLabel").textContent = p;
  const res = await getJSON(`/map?view=${state.view}&month=${p}`);
  state.values = res.values;
  if (state.view === "india_states") state.ranks = { pred: rankPct(res.values, "pred"), obs: rankPct(res.values, "obs") };
  if (geoLayer) geoLayer.setStyle(style);
  renderLegend();
  if (selected) showDetails(selected);
}

function fmt(v, d = 1) { return v === null || v === undefined ? "n/a" : Number(v).toFixed(d); }

async function showDetails(id) {
  const p = period();
  const month = state.view === "india_states" ? `${p}-01` : p;
  const panel = document.getElementById("panel");
  try {
    const r = await getJSON(`/risk?region_id=${encodeURIComponent(id)}&month=${month}`);
    let html = `<h2>${r.name || id}</h2><div class="muted">${id} &middot; ${state.view === "india_states" ? r.year : r.month}</div>`;
    if (r.view === "india_states") {
      html += `<table>
        <tr><td>Modeled incidence (per 100k, year)</td><td>${fmt(r.predicted_incidence_per_100k_year)}</td></tr>
        <tr><td>Reported incidence (per 100k, year)</td><td>${fmt(r.reported_incidence_per_100k_year)}</td></tr>
        <tr><td>Reported cases</td><td>${r.reported_cases}</td></tr></table>
        <p class="muted">${r.note}</p>`;
    } else {
      html += `<div class="big">${fmt(r.predicted_incidence_per_100k)} <span class="muted" style="font-size:13px">modeled cases per 100k</span></div>
        <div>${r.interval_level}: ${fmt(r.interval_per_100k[0])} to ${fmt(r.interval_per_100k[1])}</div>`;
      if (r.view === "brazil") {
        html += `<div>Reported: ${r.observed_incidence_per_100k === null ? "not available for this month" : fmt(r.observed_incidence_per_100k) + " per 100k (" + r.observed_cases + " cases)"}</div>
          <div class="muted">Prediction source: ${r.prediction_source}</div>`;
      } else {
        html += `<div>Relative risk: ${fmt(r.relative_risk_percentile)}th percentile among Indian districts this month</div>`;
      }
      html += `<h3 style="font-size:14px;margin:12px 0 2px">Top drivers</h3><table>`;
      for (const d of r.top_drivers) {
        html += `<tr><td>${d.label}</td><td class="${d.shap > 0 ? "up" : "down"}">${d.direction} (${d.shap > 0 ? "+" : ""}${d.shap.toFixed(2)})</td></tr>`;
      }
      html += `</table><p class="muted">${r.drivers_note}</p>`;
    }
    const m = r.data_dates;
    html += `<p class="muted">Data dates: climate through ${m.climate_data_through}, Brazil cases through ${m.brazil_cases_through}, India cases through ${m.india_cases_through}. Generated ${m.generated_utc}.</p>`;
    html += `<p class="muted"><b>Note:</b> ${r.disclaimer}</p>`;
    panel.innerHTML = html;
  } catch (e) {
    panel.innerHTML = `<h2>${id}</h2><p class="muted">No prediction for ${month}.</p>`;
  }
}

async function setView(view) {
  state.view = view;
  selected = null;
  const cfg = VIEWS[view];
  const layerSel = document.getElementById("layer");
  layerSel.innerHTML = cfg.layers.map(([k, t]) => `<option value="${k}">${t}</option>`).join("");
  state.layer = cfg.layers[0][0];
  const res = await getJSON(`/months?view=${view}`);
  state.periods = (res.months || res.years).map(String);
  const slider = document.getElementById("month");
  slider.max = state.periods.length - 1;
  // Default: latest month that has reported data in Brazil, latest period elsewhere.
  let idx = state.periods.length - 1;
  if (view === "brazil" && state.meta) {
    const j = state.periods.indexOf(state.meta.brazil_cases_through);
    if (j >= 0) idx = j;
  }
  slider.value = idx;
  document.getElementById("banner").textContent = state.meta.disclaimers[view];
  map.setView(cfg.center, cfg.zoom);
  await loadValues();
  await loadGeo(view);
  document.getElementById("panel").innerHTML = `<p class="hint">Click a region to see its predicted risk, uncertainty interval, top drivers and data dates.</p>`;
}

async function init() {
  state.meta = await getJSON("/meta");
  const m = state.meta;
  document.getElementById("footer").textContent =
    `Climate data through ${m.climate_data_through}. Latest modeled month ${m.latest_prediction_month}. ` +
    `Brazil cases (InfoDengue) through ${m.brazil_cases_through}. India state cases (OpenDengue/NCVBDC) through ${m.india_cases_through}. ` +
    `Research prototype, not a validated case forecast.`;
  document.getElementById("view").addEventListener("change", (e) => setView(e.target.value));
  document.getElementById("layer").addEventListener("change", (e) => { state.layer = e.target.value; geoLayer.setStyle(style); renderLegend(); });
  document.getElementById("month").addEventListener("input", () => loadValues());
  await setView("brazil");
}

init().catch((e) => { document.getElementById("panel").innerHTML = `<p>Failed to load: ${e.message}</p>`; });
